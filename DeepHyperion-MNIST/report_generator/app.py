# Entry point for the data

# Probably not the best way of doing it?
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import click
import logging as log
import numpy as np
import os
import json

# Avoid the Gdk-CRITICAL **: 15:09:12.737: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from functools import partial

from pathlib import Path

from report_generator.illumination_map import IlluminationAxisDefinition, IlluminationMap, select_samples_by_elapsed_time
from report_generator.samples_extractor import Sample
from report_generator.samples_extractor import DeepHyperionSample

# This introduces dependencies on cairo
#

def _store_figures_to_folder(figures, tags, run_folder):
    # figures, report
    # "Tool": "DLFuzz",
    #     "Run ID": "1",
    #     "Total Samples": 1098,
    file_format = 'pdf'

    for figure in figures:
        file_name_tokens = [figure.store_to]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        figure_file_name = "-".join(file_name_tokens) + "." + file_format

        figure_file = os.path.join(run_folder, figure_file_name)
        log.debug("Storing figure to file %s ", figure_file)
        figure.savefig(figure_file, format=file_format)


def _store_maps_to_folder(maps, tags, run_folder):
    file_format = 'npy'
    # np.save('test3.npy', a)
    for the_map in maps:
        file_name_tokens = [the_map["store_to"]]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        map_file_name = "-".join(file_name_tokens) + "." + file_format

        map_file = os.path.join(run_folder, map_file_name)
        log.debug("Storing map %s to file %s ", id, map_file)
        # Finally store it in  platform-independent numpy format
        np.save(map_file, the_map["data"])


def _store_report_to_folder(report, tags, run_folder):
    """
    Store the content of the report dict as json in the given run_folder as stats.json

    Args:
        report:
        run_folder:

    Returns:

    """
    # Basic format
    file_name_tokens = [str(report["Tool"]), str(report["Run ID"]).zfill(3)]

    # Add tags if any
    if tags is not None:
        file_name_tokens.extend([str(t) for t in tags])

    # This is the actual file name
    file_name_tokens.append("stats")

    # Add File extension
    report_file_name = "-".join(file_name_tokens) + "." + "json"

    report_file = os.path.join(run_folder, report_file_name)

    log.debug("Storing report %s to file %s ", id, report_file)
    with open(report_file, 'w') as output_file:
        output_file.writelines(json.dumps(report, indent=4))


def _log_exception(extype, value, trace):
    log.exception('Uncaught exception:', exc_info=(extype, value, trace))


def _set_up_logging(log_to, debug):
    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Process Started"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    # Configure default logging for uncaught exceptions
    sys.excepthook = _log_exception

    log.info(start_msg)


def _basic_feature_stats():
    return {
        'min' : np.PINF,
        'max' : np.NINF,
        'missing' : 0
    }


@click.group()
@click.option('--log-to', required=False, type=click.Path(exists=False), help="File to Log to. If not specified logs will show only on the console")
@click.option('--debug', required=False, is_flag=True, default=False, help="Activate debugging (more logging)")
@click.option('--show-progress', required=False, is_flag=True, default=False, help="Show some progress during the execution")
@click.pass_context
def cli(ctx, log_to, debug, show_progress):
    """
    Main entry point for the CLI. This is mostly to setup general configurations such as the logging
    """
    _set_up_logging(log_to, debug)

    # See: https://click.palletsprojects.com/en/7.x/commands/
    # Nested Commands
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['show-progress'] = show_progress


@cli.command()
@click.option('--force-attribute', type=(str, str), required=False, multiple=True, help="Attributes of the samples can be manually forced using this option.")
@click.option('--filter-samples-by-tshd', default=(None, None), type=(str, str), help="Specify a filter by name and parameters.")
@click.option('--asfault-n-sectors', required=False, default=6, type=int, help="Specify the max number of sectors per AsFatul.")
@click.argument('dataset-folder', type=click.Path(exists=True))
@click.pass_context
def generate_samples(ctx, force_attribute, filter_samples_by_tshd, asfault_n_sectors, dataset_folder):
    """
    The filter attribute is meant to skip the generation of the sample files, however, we need to be sure that
    when we create the maps we are not considering invalid samples. So if the filter triggers, we still generate
    the into...json file, but add the attributes: "invalid" e "according to filter"
    Args:
        ctx:
        force_attribute:
        filter_samples_by:
        dataset_folder:

    Returns:

    """

    # --filter-samples-by-tshd applies only to MNIST
    mnist_filter = None
    if filter_samples_by_tshd[0] is not None:
        # Make sure the import is done only if needed, this code is problematic if the libraries are not installed
        if ctx.obj['show-progress']:
            print("Filtering samples by TSHD, %s, %s" % (filter_samples_by_tshd))
            from report_generator.tshd_selection import is_valid_digit
            # Use the is_valid_digit and the input to create a partial function to filter the mnist inputs.
            # the partial function accepts now only ONE input, the sample, and is configured automatically with type
            # and val
            # The lambda makes sure that we apply partial correctly
            mnist_filter = partial( lambda tshd_type, tshd_val, sample :
                                                is_valid_digit(sample, tshd_type, tshd_val),
                                    str(filter_samples_by_tshd[0]), float(filter_samples_by_tshd[1]))
            # Make sure we record the filter setting so we can store it inside th sample file
            setattr(mnist_filter, "filter_name", "-".join(["tshd",
                                                           str(filter_samples_by_tshd[0]), str(filter_samples_by_tshd[1])]))

    # Setup the generation. Select the
    force_tool = None
    for attribute_name, attribute_value in force_attribute:

        log.debug("Forcing attribute %s with value %s", attribute_name, attribute_value)
        if attribute_name == "tool":
            force_tool = attribute_value

    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        if ctx.obj['show-progress']:
            print("Processing directory: ", subdir)

        # Consider only the files that match the pattern
        for sample_file in [os.path.join(subdir, f) for f in files if
            # TODO This is bit hacky to list all the json files that should not match...
                            not (
                                    f.startswith("info_") or
                                    f.startswith("stats") or
                                    f.startswith("config") or
                                    f.startswith("report")
                            ) and (
                                    f.endswith(".json") or
                                    f.endswith("simulation.json") or
                                    f.endswith("simulation.full.json")
                            )]:

            # TODO Replace with a factory pattern: ?
            try:


                # Despite all the effort, there are still simulation files that are empty. In this case,
                # try to use the bkp file

                # If the sample file is empty we need to look for the backup
                if not (os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0):
                    log.warning("Original file %s is empty. Try to recover with the bkp %s", sample_file, sample_file + ".bkp")
                    sample_file = sample_file + ".bkp"

                # Read the file into a
                with open(sample_file, 'r') as input_file:


                    sample_data = json.load(input_file)

                    samples = []
                    cast_as = force_tool if force_tool is not None else sample_data["tool"]

                    if cast_as == "DLFuzz":
                        sample = DLFuzzSample(os.path.splitext(sample_file)[0])
                        # add validity and re-dump if necessaty
                        if mnist_filter is not None:
                            is_valid = mnist_filter(sample)
                            sample.is_valid = is_valid
                            sample.valid_according_to = mnist_filter.filter_name
                            # We need this because dump is done automatically in the constructor,
                            # BEFORE we can check for validity
                            sample.dump()

                        samples.append(sample)

                    elif cast_as == "DeepJanus":
                        sample = DeepJanusSample(os.path.splitext(sample_file)[0])
                        # add validity and re-dump if necessaty
                        if mnist_filter is not None:
                            is_valid = mnist_filter(sample)
                            sample.is_valid = is_valid
                            sample.valid_according_to = mnist_filter.filter_name
                            # We need this because dump is done automatically in the constructor,
                            # BEFORE we can check for validity
                            sample.dump()
                        samples.append(sample)

                    elif cast_as == "DeepHyperion":
                        sample = DeepHyperionSample(os.path.splitext(sample_file)[0])
                        # add validity and re-dump if necessaty
                        if mnist_filter is not None:
                            is_valid = mnist_filter(sample)
                            sample.is_valid = is_valid
                            sample.valid_according_to = mnist_filter.filter_name
                            # We need this because dump is done automatically in the constructor,
                            # BEFORE we can check for validity
                            sample.dump()
                        samples.append(sample)

                    else:
                        log.info("Tool data is missing. Skip sample")

                # Force the value of any attribute that as been specified via "force_attribute"
                for sample in [s for s in samples if s is not None]:
                    if ctx.obj['debug']:
                        log.debug("Processing sample %s", sample_file)
                        if not sample.is_valid:
                            log.debug("*\t Sample is NOT valid according to %s", sample.valid_according_to)
                    elif ctx.obj['show-progress']:
                        if sample.is_valid:
                            # Report valid samples with an .
                            print(".", end='', flush=True)
                        else:
                            # Report invalid samples with an x
                            print("x", end='', flush=True)

                    for attribute_name, attribute_value in force_attribute:
                        setattr(sample, attribute_name, attribute_value)
                        # Make sure we dump it again
                        sample.dump()

            except Exception:
                log.warning("Error while reading file %s in main loop", sample_file, exc_info=True)
        if ctx.obj['show-progress']:
            print("")

@cli.command()
@click.option('--report-missing-features', required=False, is_flag=True, default=False, help="List all the samples that lack at least one feature")
@click.option('--parsable', required=False, is_flag=True, default=False, help="Output the stats in a parsable way")
@click.option('--feature', multiple=True, required=True, type=str, help="The name of the feature as it appears in the samples")
@click.argument('dataset-folder', nargs=-1, type=click.Path(exists=True))
@click.pass_context
def extract_stats(ctx, report_missing_features, parsable, feature, dataset_folder):
    """

    Args:
        feature: the list of name of the features to consider during the analysis

        dataset_folder: this is the root folder that contains all the results (i.e., samples as json) from all the
        tools considered in one analysis (e.g., MNIST, BeamNG)

        parsable: a flag to generate an easy to parse data

    Returns:
        for each feature report a set of descriptive values that can be computed iteratively, including total
        amount of samples, total number of samples missing a feature (sanity check), and max and min values.
        Do not consider "invalid" samples (sample.is_valid == False)

    """

    # Iteratively walk in the dataset_folder and process all the json files. For each of them compute the statistics
    data = {}
    # Overall Count
    data['total'] = 0
    # Features
    data['features'] = {f: _basic_feature_stats() for f in feature}

    # Allow multiple locations to collect samples from, as normally each tools has its own log/output folder
    # with click.progressbar(dataset_folder, label='READING') as progress_bar:
    for folder in dataset_folder:
        if ctx.obj['show-progress']:
            print("Processing directory: ", folder)

        for subdir, dirs, files in os.walk(folder, followlinks=False):
            # Consider only the files that match the pattern
            for sample_file in [os.path.join(subdir, f) for f in files if f.startswith("info_") and f.endswith(".json")]:
                log.debug("Processing sample file %s", sample_file)
                try:
                    # Read the file into a Sample, extract the feature data
                    with open(sample_file, 'r') as input_file:
                        sample_data = json.load(input_file)

                        if "is_valid" in sample_data.keys() and not sample_data["is_valid"]:
                            log.debug("Sample is not valid according to %s", sample_data["valid_according_to"])
                            continue

                        # Total count
                        data['total'] += 1

                        # Process only the that are in the sample
                        for k, v in data['features'].items():
                            # TODO There must be a pythonic way of doing it
                            if k not in sample_data["features"].keys():
                                v['missing'] += 1

                                if report_missing_features:
                                    log.warning("Sample %s miss feature %s", sample_data["id"], k)

                                continue

                            v['min'] = min(v['min'], sample_data["features"][k])
                            v['max'] = max(v['max'], sample_data["features"][k])
                except Exception:
                    log.warning("Error while reading file %s in main loop", exc_info=True)

    # Finally output the result of the analysis
    if parsable:
        for feature_name, feature_extrema in data['features'].items():
            parsable_string_tokens = ["=".join(["name",feature_name])]
            for extremum_name, extremum_value in feature_extrema.items():
                parsable_string_tokens.append("=".join([extremum_name, str(extremum_value)]))
            print(",".join(parsable_string_tokens))
    else:
        print(json.dumps(data, indent=4))


@cli.command()
@click.option('--visualize', required=False, is_flag=True, default=False, help="Visualize the generated map")
@click.option('--drop-outliers', required=False, is_flag=True, default=False, help="Drops samples that fall outside")
@click.option('--feature', multiple=True, required=True, type=(str,float,float, int), help="The description of a feature to include in the map: name, min, max, num_cell")
@click.option('--tag', required=False, type=str, multiple=True, help="A tag to add to the artifacts produced by this script")
# TODO Add validation n > 0
@click.option('--at', required=False, type=int, multiple=True, help="Specify how to select samples every N minutes")
@click.argument('run-folder', nargs=1, type=click.Path(exists=True))
@click.pass_context
def generate_map(ctx, visualize, drop_outliers, feature, tag, at, run_folder):
    """
    This command will repeat the analysis for every X minuted provided using the every option
    Args:
        visualize:
        feature:
        tag:
        at: Specify the interval at which generate the maps (--at 15, means generate the maps using samples collected
            in the first 15 minutes)
        run_folder:

    Returns:

    """
    # Generate the map axes
    map_features = []
    for f in feature:
        if ctx.obj['show-progress']:
            print("Using feature %s" % f[0])

        map_features.append(IlluminationAxisDefinition(f[0], f[1], f[2], f[3]))

    tags = []
    for t in tag:
        # TODO Leave Tags Case Sensitive
        t = str(t)
        if t not in tags:
            if ctx.obj['show-progress']:
                print("Tagging using %s" % t)
            tags.append(t)

    at_minutes = []
    for e in at:
        if e not in at_minutes:
            if ctx.obj['show-progress']:
                print("Generating Maps for samples at %s minutes" % e)
            at_minutes.append(e)

    # Note the None identify the "use all the samples" setting. By default is always the last one
    at_minutes.append(None)

    # Load all the samples, no matter what, including invalid samples and outliers
    samples = []
    for subdir, dirs, files in os.walk(run_folder, followlinks=False):
        # Consider only the files that match the pattern
        for sample_file in [os.path.join(subdir, f) for f in files if f.startswith("info_") and f.endswith(".json")]:
            log.debug("Processing sample file %s", sample_file)
            try:
                # Read the file into a Sample, extract the feature data
                with open(sample_file, 'r') as input_file:
                    samples.append(Sample.from_dict(json.load(input_file)))
            except Exception:
                log.warning("Error while reading file %s in main loop", exc_info=True)

    the_map = IlluminationMap(map_features, set(samples), drop_outliers=drop_outliers)

    for e in at_minutes:

        select_samples = None
        _tags = tags[:]
        if e is not None:
            if ctx.obj['show-progress']:
                print("Selecting samples within ", e, "minutes")
            # Create the sample selector
            select_samples = select_samples_by_elapsed_time(e)

            # Add the minutes ot the tag, create a copy so we do not break the given set of tags
            _tags.append("".join([str(e).zfill(2), "min"]))
        else:
            if ctx.obj['show-progress']:
                print("Selecting all the samples")

        report = the_map.compute_statistics(tags=_tags, sample_selector=select_samples)

        # Show this if debug is enabled
        log.debug(json.dumps(report, indent=4))

        # Store the report as json
        _store_report_to_folder(report, _tags, run_folder)

        # Create coverage and probability figures
        figures = the_map.visualize(tags=_tags, sample_selector=select_samples)
        # store figures without showing them
        _store_figures_to_folder(figures, _tags, run_folder)

        figures, probability_maps, misbehaviour_maps, coverage_maps = the_map.visualize_probability(tags=_tags, sample_selector=select_samples)
        # store the outputs
        _store_figures_to_folder(figures, _tags, run_folder)
        # store maps
        _store_maps_to_folder(probability_maps, _tags, run_folder)
        _store_maps_to_folder(misbehaviour_maps, _tags, run_folder)
        _store_maps_to_folder(coverage_maps, _tags, run_folder)

    # Visualize Everything at the end
    if visualize:
        plt.show()
    else:
        # Close all the figures if open
        for figure in figures:
            plt.close(figure)


# Invoked the CLI
if __name__ == '__main__':
    cli()



