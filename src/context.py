"""Build source-only context without crossing chapter boundaries."""

from dataclasses import dataclass
from collections.abc import Sequence

from src.dataset import Sentence

CONTEXT_SIZES = (0, 1, 2, 3)


@dataclass(frozen=True)
class ContextExample:
    sentence: Sentence
    context_size: int
    previous_sources: tuple[str, ...]

    @property
    def input_text(self) -> str:
        """Inspectable formatting; model-specific formatting comes later.

        References never enter the model input. A future inference adapter must
        translate only CURRENT, using PREVIOUS as context rather than output.
        """
        previous = "\n".join(self.previous_sources)
        return f"PREVIOUS:\n{previous}\n\nCURRENT:\n{self.sentence.source}"


def build_context(
    sentences: Sequence[Sentence], sentence_index: int, context_size: int
) -> ContextExample:
    if context_size not in CONTEXT_SIZES:
        raise ValueError("context_size must be 0, 1, 2, or 3")
    if not 0 <= sentence_index < len(sentences):
        raise IndexError("sentence_index is outside the dataset")
    current = sentences[sentence_index]
    previous = []
    for index in range(sentence_index - 1, -1, -1):
        if len(previous) == context_size or sentences[index].chapter_id != current.chapter_id:
            break
        previous.append(sentences[index].source)
    return ContextExample(current, context_size, tuple(reversed(previous)))
