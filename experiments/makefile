DATA=data
BEAMNG_DATA=$(DATA)/beamng
MNIST_DATA=$(DATA)/mnist
PYTHON=.venv/bin/python
APP=data_analysis.py

### SETUP PYTHON VENV


.venv:	.venv/touchfile


.venv/touchfile: requirements.txt
	test -d .venv || python3 -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	touch .venv/touchfile

### DOWNLOAD EXPERIMENTAL DATA

$(BEAMNG_DATA):
	mkdir -p $(BEAMNG_DATA) && \
	tar -xzvf $(DATA)/beamng.tar.gz --directory $(BEAMNG_DATA) || \
	$(RM) -v $(DATA)/beamng.tar.gz


$(MNIST_DATA):
	mkdir -p $(MNIST_DATA) && \
	tar -xzvf $(DATA)/mnist.tar.gz --directory $(MNIST_DATA) || \
	$(RM) -v $(DATA)/mnist.tar.gz

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

.PHONY: rq1 rq2 clean-beamng-data clean-mnist-data

# Plots and stats
rq1 rq2 rq3: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(PYTHON) $(APP) $@ 2>&1 | tee $@.log


./plots/RQ1-MNIST.pdf ./plots/RQ1-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(PYTHON) rq1.py

./plots/RQ2-MNIST.pdf ./plots/RQ2-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(PYTHON) rq2.py

./plots/RQ3-MNIST.pdf ./plots/RQ3-BeamNG.pdf: $(BEAMNG_DATA) $(MNIST_DATA) .venv
	$(PYTHON) rq3.py

final-plots: ./plots/RQ1-MNIST.pdf ./plots/RQ1-BeamNG.pdf ./plots/RQ2-MNIST.pdf ./plots/RQ2-BeamNG.pdf ./plots/RQ3-MNIST.pdf ./plots/RQ3-BeamNG.pdf
	@echo Done
