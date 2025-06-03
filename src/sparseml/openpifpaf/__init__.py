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

import logging
from sparseml.analytics import sparseml_analytics as _analytics


_LOGGER = logging.getLogger(__name__)

try:
    import cv2 as _cv2  # noqa: F401

    import openpifpaf as _openpifpaf  # noqa: F401
except ImportError as err:
    _LOGGER.exception(
        "Please install sparseml[openpifpaf] to use this pathway", exc_info=True
    )
    raise ImportError(
        "Please install sparseml[openpifpaf] to use this pathway"
    ) from err


_analytics.send_event("python__openpifpaf__init")
