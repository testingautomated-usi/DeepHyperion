# Global variables
SHELL=/bin/bash
DATA=data
BEAMNG_DATA=$(DATA)/beamng
MNIST_DATA=$(DATA)/mnist
V_PYTHON=.venv/bin/python
APP=data_analysis.py

# Check that python 3.7 is installed. See: https://stackoverflow.com/questions/4933285/how-to-detemine-python-version-in-makefile

# Manually define anther PYTHON here if needed
PYTHON3=python3

PYTHON3_OK := $(shell type -P $(PYTHON3))
ifeq ('$(PYTHON3_OK)','')
    $(error package '$(PYTHON3)' not found)
endif

PYTHON_VERSION_MIN=3.6
PYTHON_VERSION_MAX=3.8

PYTHON_VERSION=$(shell $(PYTHON3) -c 'import sys; print("%d.%d"% sys.version_info[0:2])' )
PYTHON_VERSION_OK=$(shell $(PYTHON3) -c 'import sys;\
  print(int(float("%d.%d"% sys.version_info[0:2]) >= $(PYTHON_VERSION_MIN) and float("%d.%d"% sys.version_info[0:2]) < $(PYTHON_VERSION_MAX)))' )

ifeq ($(PYTHON_VERSION_OK),0)
  $(error "Need python $(PYTHON_VERSION_MIN) but found $(PYTHON_VERSION)")
endif

### Targets definition

.PHONY: rq1 rq2 rq3 clean-beamng-data clean-mnist-data

### Setup Python virtual environment

.venv:	.venv/touchfile

.venv/touchfile: requirements.txt
	test -d .venv || $(PYTHON3) -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	touch .venv/touchfile

### Prepare the data EXPERIMENTAL DATA

$(BEAMNG_DATA):
	mkdir -p $(BEAMNG_DATA) && \
	tar -xzvf $(DATA)/beamng.tar.gz --directory $(BEAMNG_DATA)

$(MNIST_DATA):
	mkdir -p $(MNIST_DATA) && \
	tar -xzvf $(DATA)/mnist.tar.gz --directory $(MNIST_DATA)

### Clean up

clean-beamng-data:
	$(RM) -r -v $(BEAMNG_DATA)

clean-mnist-data:
	$(RM) -r -v $(MNIST_DATA)

clean-data: clean-mnist-data clean-beamng-data
	@echo "Done"

clean-plots: 
	$(RM) -v plots/*.pdf
	$(RM) -v plots/*.stats
	$(RM) -v plots/*.npy

### Plotting Targets. 

./plots/RQ1-MNIST.pdf ./plots/RQ1-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq1.py

./plots/RQ2-MNIST.pdf ./plots/RQ2-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq2.py

./plots/RQ3-MNIST.pdf ./plots/RQ3-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq3.py

### Plotting "Commands". Use the following only if you want to plot each single figure, otherwise use plot-all

rq1: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq1.py 2>&1 | tee $@.log

rq2: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq2.py 2>&1 | tee $@.log

rq3: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(V_PYTHON) rq3.py 2>&1 | tee $@.log

### Plot all the figures in the paper from the data

plot-all: ./plots/RQ1-MNIST.pdf ./plots/RQ1-BeamNG.pdf ./plots/RQ2-MNIST.pdf ./plots/RQ2-BeamNG.pdf ./plots/RQ3-MNIST.pdf ./plots/RQ3-BeamNG.pdf
	@echo Done
