"""Training stage implementations: LoRA setup, CPT, and SFT trainers.

Heavy GPU dependencies (unsloth, bitsandbytes, torch) are imported lazily
inside functions/classes rather than at module import time, so that this
package can be imported (e.g. for type checking or by the API layer) on
machines without a GPU or those libraries installed.
"""