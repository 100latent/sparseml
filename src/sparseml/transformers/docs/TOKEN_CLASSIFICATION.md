# SparseML Transformer Token Classification Integration


SparseML Token Classification pipeline integrates with Hugging Face’s Transformers library to enable the sparsification of a large set of transformers models.
Sparsification is a powerful technique that results in faster, smaller, and cheaper deployable models. 
After training, the model can be deployed with Neural Magic's DeepSparse Engine. The engine enables inference with GPU-class performance directly on your CPU.

This integration enables spinning up one of the following end-to-end functionalities:
- **Sparsification of Popular Transformer Models** - easily sparsify any popular Hugging Face Transformer model. 
- **Sparse Transfer Learning** - fine-tune a sparse backbone model (or use one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=token_classification)) on your own private dataset.

## Installation

```pip install sparseml[torch]```

It is recommended to run Python 3.8 as some of the scripts within the transformers repository require it.

Note: Transformers will not immediately install with this command. Instead, a sparsification-compatible version of Transformers will install on the first invocation of the Transformers code in SparseML.

## Getting Started

### Sparsifying Popular Transformer Models


In the example below, a dense BERT model is trained on the CoNLL-2003 dataset. By passing the recipe `zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni` (located in [SparseZoo](https://sparsezoo.neuralmagic.com/models/nlp%2Ftoken_classification%2Fbert-base%2Fpytorch%2Fhuggingface%2Fconll2003%2F12layer_pruned80_quant-none-vnni)) we modify (sparsify) the training process and/or the model.

```bash
sparseml.transformers.token_classification \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --do_train \
  --do_eval \
  --output_dir './output' \
  --cache_dir cache \
  --distill_teacher disable \
  --recipe zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni        
```

### Sparse Transfer Learning

Once you sparsify a model using SparseML, you can easily sparse fine-tune it on a new dataset.
While you are free to use your backbone, we encourage you to leverage one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com) to boost your productivity!

In the example below, we fetch a pruned, quantized BERT model, pre-trained on Wikipedia and Bookcorpus datasets. We then fine-tune the model to the CoNLL-2003 dataset. 
```bash
sparseml.transformers.token_classification \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni \
    --dataset_name conll2003 \
    --do_train \
    --do_eval \
    --output_dir './output' \
    --distill_teacher disable \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-token_classification
```

#### Knowledge Distillation
By modifying the `distill_teacher` argument, you can enable [Knowledge Distillation](https://neptune.ai/blog/knowledge-distillation) (KD) functionality.

In this example, the `--distill_teacher` argument is set to pull a dense CoNLL-2003 model from the SparseZoo to enable it to run independently of the dense teacher step:

```bash
--distill_teacher zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/base-none 
```

Alternatively, the user may decide to train their own dense teacher model. The following command uses the dense BERT base model from the SparseZoo and fine-tunes it on the CoNLL-2003 dataset.
```bash
sparseml.transformers.token_classification \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
    --dataset_name conll2003 \
    --do_train \
    --do_eval \
    --output_dir models/teacher \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-token_classification 
```

Once the dense teacher is trained we may reuse it for KD. This only requires passing the path to the directory with the model:

```bash
--distill_teacher models/teacher
```

## SparseML CLI

The SparseML installation provides a CLI for sparsifying your models for a specific task; appending the `--help` argument displays a full list of options for training in SparseML:
```bash
sparseml.transformers.token_classification --help
```
output:
```bash
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model, sparsezoo stub. or model identifier from huggingface.co/models (default: None)
  --distill_teacher DISTILL_TEACHER
                        Teacher model which needs to be a trained NER model (default: None)
  --cache_dir CACHE_DIR
                        Where to store the pretrained data from huggingface.co (default: None)
  --recipe RECIPE       
                        Path to a SparseML sparsification recipe, see https://github.com/neuralmagic/sparseml for more information (default: None)
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library) (default: None)
  ...
```

To learn about the Hugging Face Transformers parameters in more detail, refer to [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

## Once the Training is Done...

The artifacts of the training process are saved to the directory `--output_dir`. Once the script terminates, the directory will have everything required to deploy or further modify the model such as:
- The recipe (with the full description of the sparsification attributes).
- Checkpoint files (saved in the appropriate framework format).
- Additional configuration files (e.g., tokenizer, dataset info).


### Exporting the Sparse Model to ONNX

The DeepSparse Engine uses the ONNX format to load neural networks and then deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.transformers.export_onnx` command that you can use to load the training model folder and create a new `model.onnx` file within. Be sure the `--model_path` argument points to your trained model. 
```bash
sparseml.transformers.export_onnx \
    --model_path './output' \
    --task 'token-classification' 
```

### DeepSparse Engine Deployment

Once the model is exported in the ONNX format, it is ready for deployment with the DeepSparse Engine. 

The deployment is intuitive due to the DeepSparse Python API.

```python
from deepsparse import Pipeline

tc_pipeline = Pipeline.create(
  task="token-classification", 
  model_path='./output'
)
inference = tc_pipeline("We are flying from Texas to California")
>> [{'entity': 'LABEL_0', 'word': 'we', ...}, 
    {'entity': 'LABEL_0', 'word': 'are', ...}, 
    {'entity': 'LABEL_0', 'word': 'flying', ...}, 
    {'entity': 'LABEL_0', 'word': 'from', ...}, 
    {'entity': 'LABEL_5', 'word': 'texas', ...}, 
    {'entity': 'LABEL_0', 'word': 'to', ...}, 
    {'entity': 'LABEL_5', 'word': 'california', ...}]
```


To learn more, refer to the [appropriate documentation in the DeepSparse repository](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/README.md).

## Support

For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/sparseml/issues).
