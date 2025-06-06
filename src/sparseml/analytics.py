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

from sparseml.version import version as sparseml_version

try:
    from sparsezoo.analytics import GoogleAnalytics
except Exception:  # pragma: no cover - optional dependency stub
    class GoogleAnalytics:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def send_event(self, *args, **kwargs):
            pass


__all__ = ["sparseml_analytics"]


# analytics client for sparseml, to disable set NM_DISABLE_ANALYTICS=1
sparseml_analytics = GoogleAnalytics("sparseml", sparseml_version)
