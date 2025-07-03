# 🚀 Add MinifiedPydanticOutputParser to Optimize Token Usage in LLM Outputs

Here's a new class: MinifiedPydanticOutputParser, a drop-in replacement for PydanticOutputParser that automatically reduces token usage by replacing field names with short identifiers (e.g., a, b, c, …), while preserving full descriptions and reversibility.

![MinifiedPydanticOutputParser](https://github.com/JulienStitelet/langchain-pydantic-minifier/blob/main/resources/splash.png)

## ✨ What It Does
* Transforms a given Pydantic schema by replacing verbose field names with shorter aliases.
* Retains all Field(..., description=...) information — essential for prompt construction and LLM understanding.
* Accepts minified JSON outputs from the LLM and reconstructs the original schema transparently.
* Supports nested models and list fields recursively.
* Compatible with strict=True mode used with with_structured_output.

## ✅ Benefits
* Reduces prompt and completion token count, leading to faster LLM response times and lower inference costs.
* Maintains clarity in the LLM's understanding of the field semantics thanks to preserved descriptions.
* No code change required downstream — consumers receive the original schema post-parsing.

## 📉 Performance Impact
In personal benchmarks, this optimization led to a ~30% reduction in LLM response time, due to:
* Fewer tokens needing generation
* Reduced I/O and parsing overhead

## 💸 This also translates to lower API costs, especially in high-throughput or large-output scenarios.

## Examples

* [See Gemini example](https://github.com/JulienStitelet/langchain-pydantic-minifier/blob/main/examples/test_minified_gemini_with_parser.md)
* [see OpenAI OutputParser example](https://github.com/JulienStitelet/langchain-pydantic-minifier/blob/main/examples/openai_output_parser.md)
* [see OpenAI structured_output strict example](https://github.com/JulienStitelet/langchain-pydantic-minifier/blob/main/examples/test_minified_with_structured_output.md)


## 🧪 What it does

Given:
```python
class User(BaseModel):
    first_name: str = Field(..., description="The user's first name")
    last_name: str = Field(..., description="The user's last name")
```
The model is transformed into:
```python
class MinifiedUser(BaseModel):
    a: str = Field(..., description="The user's first name")
    b: str = Field(..., description="The user's last name")
```

Then seamlessly restored to the original User class after parsing the LLM output.