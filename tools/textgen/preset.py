from .constants import DEFAULT_STYLE_INFO, QUESTION_DEFAULT_STYLE_ADDON


class Preset:
  def __init__(
    self, 
    lang: str, 
    batch_size: int, 
    max_length: int | None = None,
    style: str | None = None
  ):
    self.lang = lang
    self.batch_size = batch_size
    self.max_length = max_length
    self.style = style


class StatementPreset(Preset):
  def __init__(self, lang: str, batch_size: int, max_length: int | None = None, style: str | None = None):
    super().__init__(lang, batch_size, max_length, style)

    limit_info = f"\nEach statement must not exceed {self.max_length} words.\n" if self.max_length else ""

    self.style_prompt = DEFAULT_STYLE_INFO if not self.style else f"Imitate the linguistic style of the provided samples:\n{self.style}"

    self.prompt = f"""Generate {self.batch_size} diverse statement sentences in {self.lang}.{limit_info}
{self.style_prompt}

Ensure variety in statement types and length - from very short (1 or 2 words) to long, compound sentences.

Format the output as a simple list of {self.batch_size} statements, one per line.
No numbering, no extra text, just the statements."""


class QuestionPreset(Preset):
  def __init__(self, lang: str, batch_size: int, max_length: int | None = None, style: str | None = None):
    super().__init__(lang, batch_size, max_length, style)

    limit_info = f"\nEach question must not exceed {self.max_length} words.\n" if self.max_length else ""
    style_info = self.style if self.style else DEFAULT_STYLE_INFO + "\n" + QUESTION_DEFAULT_STYLE_ADDON

    self.prompt = f"""Generate {self.batch_size} diverse questions in {self.lang}.{limit_info}
{style_info}

Also ensure variety in length - from very short (1 or 2 words) to long, compound sentences.

Format the output as a simple list of {self.batch_size} questions, one per line.
No numbering, no extra text, just the questions."""


class ExclamationPreset(Preset):
  def __init__(self, lang: str, batch_size: int, max_length: int | None = None, style: str | None = None):
    super().__init__(lang, batch_size, max_length, style)

    limit_info = f"\nEach exclamation must not exceed {self.max_length} words.\n" if self.max_length else ""

    self.style_prompt = DEFAULT_STYLE_INFO if not self.style else f"Imitate the linguistic style of the provided samples:\n{self.style}"

    self.prompt = f"""Generate {self.batch_size} diverse exclamation sentences in {self.lang}.{limit_info}
{self.style_prompt}

Ensure variety in intensity and length - from single-word interjections to full, emotive sentences.

Format the output as a simple list of {self.batch_size} exclamations, one per line.
No numbering, no extra text, just the exclamations."""


class MultiSentencePreset(Preset):
  def __init__(
    self, 
    lang: str, 
    batch_size: int,
    max_length: int | None = None,
    style: str | None = None
  ):
    super().__init__(lang, batch_size, max_length, style)

    limit_info = f"\nEach text must not exceed {self.max_length} words." if self.max_length else ""

    self.style_prompt = DEFAULT_STYLE_INFO if not self.style else f"Imitate the linguistic style of the provided samples:\n{self.style}"

    self.prompt = f"""Generate {self.batch_size} diverse multi-sentence texts in {self.lang}.{limit_info}
Each text should consist of at least 2 and at most 3 short sentences.

{self.style_prompt}

Ensure variety in tone, complexity, and length across the {self.batch_size} texts.

Format the output as a simple list of {self.batch_size} texts, one per line.
No numbering, no extra text, just the multi-sentence texts."""
