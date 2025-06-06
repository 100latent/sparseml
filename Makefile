.PHONY: build docs test

BUILDDIR := $(PWD)
CHECKDIRS := integrations src tests utils status examples
CHECKGLOBS := 'integrations/**/*.py' 'src/**/*.py' 'tests/**/*.py' 'utils/**/*.py' 'status/**/*.py'
DOCDIR := docs
MDCHECKGLOBS := 'docs/**/*.md' 'docs/**/*.rst' 'integrations/**/*.md'
MDCHECKFILES := CODE_OF_CONDUCT.md CONTRIBUTING.md DEVELOPING.md README.md
SPARSEZOO_TEST_MODE := "false"

BUILD_ARGS :=  # set nightly to build nightly release
TARGETS := ""  # targets for running pytests: deepsparse,keras,onnx,pytorch,pytorch_models,export,pytorch_datasets,tensorflow_v1,tensorflow_v1_models,tensorflow_v1_datasets
PYTEST_ARGS ?= ""
PYTEST_INTEG_ARGS ?= ""
ifneq ($(findstring deepsparse,$(TARGETS)),deepsparse)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/deepsparse
endif
ifneq ($(findstring transformers,$(TARGETS)),transformers)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/transformers
endif
ifneq ($(findstring export,$(TARGETS)),export)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/export
endif
ifneq ($(findstring keras,$(TARGETS)),keras)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/keras
endif
ifneq ($(findstring onnx,$(TARGETS)),onnx)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/onnx
endif
ifneq ($(findstring pytorch,$(TARGETS)),pytorch)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/pytorch
endif
ifneq ($(findstring pytorch_models,$(TARGETS)),pytorch_models)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/pytorch/models
endif
ifneq ($(findstring pytorch_datasets,$(TARGETS)),pytorch_datasets)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/pytorch/datasets
endif
ifneq ($(findstring tensorflow_v1,$(TARGETS)),tensorflow_v1)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/tensorflow_v1
endif
ifneq ($(findstring tensorflow_v1_models,$(TARGETS)),tensorflow_v1_models)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/tensorflow_v1/models
endif
ifneq ($(findstring tensorflow_v1_datasets,$(TARGETS)),tensorflow_v1_datasets)
    PYTEST_ARGS := $(PYTEST_ARGS) --ignore tests/sparseml/tensorflow_v1/datasets
endif
ifneq ($(findstring image_classification,$(TARGETS)),image_classification)
    PYTEST_INTEGRATION_ARGS := $(PYTEST_INTEGRATION_ARGS) --ignore tests/integrations/image_classification
endif
ifneq ($(findstring transformers,$(TARGETS)),transformers)
    PYTEST_INTEGRATION_ARGS := $(PYTEST_INTEGRATION_ARGS) --ignore tests/integrations/transformers
endif
ifneq ($(findstring yolov5,$(TARGETS)),yolov5)
    PYTEST_INTEGRATION_ARGS := $(PYTEST_INTEGRATION_ARGS) --ignore tests/integrations/yolov5
endif


# run checks on all files for the repo
quality:
	@echo "Running copyright checks";
	python utils/copyright.py quality $(CHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python quality checks";
	uv run black --check $(CHECKDIRS);
	uv run isort --check-only $(CHECKDIRS);
	uv run flake8 $(CHECKDIRS);

# style the code according to accepted standards for the repo
style:
	@echo "Running copyrighting";
	uv run python utils/copyright.py style $(CHECKGLOBS) $(MDCHECKGLOBS) $(MDCHECKFILES)
	@echo "Running python styling";
	uv run black $(CHECKDIRS);
	uv run isort $(CHECKDIRS);

# run tests for the repo
test:
	@echo "Running python tests";
	SPARSEZOO_TEST_MODE="true" uv run pytest tests $(PYTEST_ARGS) --ignore tests/integrations

# run integration tests
testinteg:
	@echo "Running integration tests";
	SPARSEZOO_TEST_MODE="true" uv run pytest -x -ls tests/integrations $(PYTEST_INTEGRATION_ARGS)

# create docs
docs:
	@echo "Running docs creation";
	export SPARSEML_IGNORE_TFV1="True"; uv run utils/docs_builder.py --src $(DOCDIR) --dest $(DOCDIR)/build/html;

docsupdate:
	@echo "Runnning update to api docs";
	find $(DOCDIR)/api | grep .rst | xargs rm -rf;
	export SPARSEML_IGNORE_TFV1="True"; uv run sphinx-apidoc -o "$(DOCDIR)/api" src/sparseml;

# creates wheel file
build:
	uv build sdist bdist_wheel $(BUILD_ARGS);

# clean package
clean:
	uv clean;
	rm -fr .pytest_cache;
	rm -fr docs/_build docs/build;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;
