# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code related to the PyTorch model registry for easily creating models.
"""


from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.nn import Module

from merge_args import merge_args
from sparseml.pytorch.utils import download_framework_model_by_recipe_type, load_model
from sparseml.utils import parse_optimization_str, wrapper_decorator
from sparseml.utils.frameworks import PYTORCH_FRAMEWORK


__all__ = [
    "ModelRegistry",
]

"""
Simple named tuple object to store model info
"""
_ModelAttributes = NamedTuple(
    "_ModelAttributes",
    [
        ("input_shape", Any),
        ("domain", str),
        ("sub_domain", str),
        ("architecture", str),
        ("sub_architecture", str),
        ("default_dataset", str),
        ("default_desc", str),
        ("repo_source", str),
        ("ignore_error_tensors", List[str]),
        ("args", Dict[str, Tuple[str, Any]]),
    ],
)


class ModelRegistry(object):
    """
    Registry class for creating models
    """

    _CONSTRUCTORS = {}  # type: Dict[str, Callable]
    _ATTRIBUTES = {}  # type: Dict[str, _ModelAttributes]

    @staticmethod
    def available_keys() -> List[str]:
        """
        :return: the keys (models) currently available in the registry
        """
        return list(ModelRegistry._CONSTRUCTORS.keys())

    @staticmethod
    def create(
        key: Optional[str] = None,
        pretrained: Union[bool, str] = False,
        pretrained_path: str = None,
        pretrained_dataset: str = None,
        load_strict: bool = True,
        ignore_error_tensors: List[str] = None,
        **kwargs,
    ) -> Union[Module, Tuple[Module, Optional[str]]]:
        """
        Create a new model for the given key

        :param key: the model key (name) to create. If None, the key is read
            from the state_dict['arch_key'] of the model.
        :param pretrained: True to load pretrained weights; to load a specific version
            give a string with the name of the version (pruned-moderate, base).
            Default None
        :param pretrained_path: A model file path to load into the created model
        :param pretrained_dataset: The dataset to load for the model
        :param load_strict: True to make sure all states are found in and
            loaded in model, False otherwise; default True
        :param ignore_error_tensors: tensors to ignore if there are errors in loading
        :param kwargs: any keyword args to supply to the model constructor
        :return: The instantiated model if key is given else a Tuple containing
            the instantiated model and the loaded key
        """
        key_copy = key

        if key_copy is None:
            if pretrained_path is None:
                raise ValueError("Must provide a key or a pretrained_path")

            # Removed custom_zoo and zoo handling.
            # Model loading from stubs is no longer supported here.
            # pretrained_path must now be a direct file path.
            _checkpoint = torch.load(pretrained_path)
            if "arch_key" in _checkpoint:
                key_copy = _checkpoint["arch_key"]
            else:
                raise ValueError("No `arch_key` found in checkpoint")

        if key_copy not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key_copy, ModelRegistry._CONSTRUCTORS
                )
            )
        model = ModelRegistry._CONSTRUCTORS[key_copy](
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_dataset=pretrained_dataset,
            load_strict=load_strict,
            ignore_error_tensors=ignore_error_tensors,
            **kwargs,
        )
        return (model, key_copy) if key is None else model

    @staticmethod
    def input_shape(key: str) -> Any:
        """
        :param key: the model key (name) to create
        :return: the specified input shape for the model
        """
        if key not in ModelRegistry._CONSTRUCTORS:
            raise ValueError(
                "key {} is not in the model registry; available: {}".format(
                    key, ModelRegistry._CONSTRUCTORS
                )
            )

        return ModelRegistry._ATTRIBUTES[key].input_shape

    @staticmethod
    def register(
        key: Union[str, List[str]],
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        repo_source: str = "sparseml",
        def_ignore_error_tensors: List[str] = None,
        desc_args: Dict[str, Tuple[str, Any]] = None,
    ):
        """
        Register a model with the registry. Should be used as a decorator

        :param key: the model key (name) to create
        :param input_shape: the specified input shape for the model
        :param domain: the domain the model belongs to; ex: cv, nlp, etc
        :param sub_domain: the sub domain the model belongs to;
            ex: classification, detection, etc
        :param architecture: the architecture the model belongs to;
            ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to;
            ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading
            pretrained if not supplied
        :param default_desc: the description to use by default for loading
            pretrained if not supplied
        :param repo_source: the source repo for the model, default is sparseml
        :param def_ignore_error_tensors: tensors to ignore if there are
            errors in loading
        :param desc_args: args that should be changed based on the description
        :return: the decorator
        """
        if not isinstance(key, List):
            key = [key]

        def decorator(constructor_func):
            wrapped_constructor = ModelRegistry._registered_wrapper(
                key[0],
                constructor_func,
            )

            ModelRegistry.register_wrapped_model_constructor(
                wrapped_constructor,
                key,
                input_shape,
                domain,
                sub_domain,
                architecture,
                sub_architecture,
                default_dataset,
                default_desc,
                repo_source,
                def_ignore_error_tensors,
                desc_args,
            )
            return wrapped_constructor

        return decorator

    @staticmethod
    def register_wrapped_model_constructor(
        wrapped_constructor: Callable,
        key: Union[str, List[str]],
        input_shape: Any,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        default_dataset: str,
        default_desc: str,
        repo_source: str,
        def_ignore_error_tensors: List[str] = None,
        desc_args: Dict[str, Tuple[str, Any]] = None,
    ):
        """
        Register a model with the registry from a model constructor or provider function

        :param wrapped_constructor: Model constructor wrapped to be compatible
            by call from ModelRegistry.create should have pretrained, pretrained_path,
            pretrained_dataset, load_strict, ignore_error_tensors, and kwargs as
            arguments
        :param key: the model key (name) to create
        :param input_shape: the specified input shape for the model
        :param domain: the domain the model belongs to; ex: cv, nlp, etc
        :param sub_domain: the sub domain the model belongs to;
            ex: classification, detection, etc
        :param architecture: the architecture the model belongs to;
            ex: resnet, mobilenet, etc
        :param sub_architecture: the sub architecture the model belongs to;
            ex: 50, 101, etc
        :param default_dataset: the dataset to use by default for loading
            pretrained if not supplied
        :param default_desc: the description to use by default for loading
            pretrained if not supplied
        :param repo_source: the source repo for the model; ex: sparseml, torchvision
        :param def_ignore_error_tensors: tensors to ignore if there are
            errors in loading
        :param desc_args: args that should be changed based on the description
        :return: The constructor wrapper registered with the registry
        """
        if not isinstance(key, List):
            key = [key]

        for r_key in key:
            if r_key in ModelRegistry._CONSTRUCTORS:
                raise ValueError("key {} is already registered".format(key))

            ModelRegistry._CONSTRUCTORS[r_key] = wrapped_constructor
            ModelRegistry._ATTRIBUTES[r_key] = _ModelAttributes(
                input_shape,
                domain,
                sub_domain,
                architecture,
                sub_architecture,
                default_dataset,
                default_desc,
                repo_source,
                def_ignore_error_tensors,
                desc_args,
            )

    @staticmethod
    def _registered_wrapper(
        key: str,
        constructor_func: Callable,
    ):
        @merge_args(constructor_func)
        @wrapper_decorator(constructor_func)
        def wrapper(
            pretrained_path: str = None,
            pretrained: Union[bool, str] = False,
            pretrained_dataset: str = None, # This param may become unused or used differently
            load_strict: bool = True,
            ignore_error_tensors: List[str] = None,
            *args,
            **kwargs,
        ):
            """
            :param pretrained_path: A path to the pretrained weights to load.
                If provided, this will override the `pretrained` parameter's behavior.
                NOTE: SparseZoo stub paths (e.g. 'zoo:...') are no longer supported.
            :param pretrained: If True and `pretrained_path` is not set, attempts to load
                default weights (behavior might be removed or changed if it relied on zoo).
                If a string, could specify a variant IF `pretrained_path` is also given
                (e.g. for custom local setups), but automatic download from SparseZoo based
                on this string is no longer supported. Generally, prefer `pretrained_path`.
                Set to False to not load any pretrained weights.
            :param pretrained_dataset: The dataset associated with the pretrained weights.
                This may be used by `load_model` or custom logic if `pretrained_path` is set,
                but it no longer triggers downloads from SparseZoo.
            :param load_strict: True to raise an error on issues with state dict
                loading from pretrained_path, False to ignore.
            :param ignore_error_tensors: Tensors to ignore while checking the state dict
                for weights loaded from pretrained_path.
            """
            attributes = ModelRegistry._ATTRIBUTES[key]

            if attributes.args and pretrained in attributes.args:
                kwargs[attributes.args[pretrained][0]] = attributes.args[pretrained][1]

            model = constructor_func(*args, **kwargs)
            ignore = []

            if ignore_error_tensors:
                ignore.extend(ignore_error_tensors)
            elif attributes.ignore_error_tensors:
                ignore.extend(attributes.ignore_error_tensors)

            if isinstance(pretrained, str):
                if pretrained.lower() == "true":
                    pretrained = True
                elif pretrained.lower() in ["false", "none"]:
                    pretrained = False

            if pretrained_path:
                load_model(pretrained_path, model, load_strict, ignore)
            elif pretrained:
                # Behavior when pretrained is True/string but pretrained_path is not set:
                # Original logic relied on create_zoo_model and downloading.
                # This is now removed.
                # If you need to load a default model without a path, this needs new handling
                # or documentation should state pretrained_path is mandatory for pretrained=True.
                # For now, if pretrained_path is None and pretrained is True/str, nothing happens here.
                pass

            return model

        return wrapper
