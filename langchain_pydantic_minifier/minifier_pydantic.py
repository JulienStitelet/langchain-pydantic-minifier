"""Memory-optimized Minification tool class for Output parsers using Pydantic."""

from typing import Any, Optional, Union, get_args, get_origin

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs.generation import Generation
from pydantic import BaseModel, ConfigDict, Field


class MinifiedPydanticOutputParser(PydanticOutputParser):
    """Memory-optimized minified Pydantic schema parser."""

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        pydantic_object: type[BaseModel],
        *,
        strict: Optional[bool] = False,
    ):
        """Initialize with memory optimizations."""
        super().__init__(pydantic_object=pydantic_object)
        self.strict = strict
        self.original_type = pydantic_object

        # Use a simple counter for field names instead of pre-generating
        self._field_counter = 0
        self.field_names_mapper: dict[str, str] = {}

        self.minified = self._make_fields_required_and_small(pydantic_object)
        self.pydantic_object = self.minified

    def _get_next_short_name(self) -> str:
        """Generate short names on demand."""
        if self._field_counter < 26:
            # a-z
            name = chr(ord("a") + self._field_counter)
        elif self._field_counter < 26 + 26 * 26:
            # aa-zz
            offset = self._field_counter - 26
            first = chr(ord("a") + offset // 26)
            second = chr(ord("a") + offset % 26)
            name = first + second
        else:
            # aaa-zzz and beyond
            offset = self._field_counter - 26 - 26 * 26
            first = chr(ord("a") + offset // (26 * 26))
            second = chr(ord("a") + (offset // 26) % 26)
            third = chr(ord("a") + offset % 26)
            name = first + second + third

        self._field_counter += 1
        return name

    def _get_short_field_name(self, field_name: str) -> str:
        """Get or generate short field name."""
        if field_name not in self.field_names_mapper:
            self.field_names_mapper[field_name] = self._get_next_short_name()
        return self.field_names_mapper[field_name]

    def _make_fields_required_and_small(
        self, pydantic_cls: type[BaseModel]
    ) -> type[BaseModel]:
        """Transform fields with reduced memory allocation."""
        original_fields = pydantic_cls.model_fields

        new_annotations: dict[str, Any] = {}
        new_fields: dict[str, Any] = {}

        for name, field in original_fields.items():
            field_type = field.annotation
            short_name = self._get_short_field_name(name)

            # Handle Union types (Optional)
            if get_origin(field_type) is Union:
                non_none_args = [
                    arg for arg in get_args(field_type) if arg is not type(None)
                ]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]

            # Handle nested BaseModel
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_type = self._make_fields_required_and_small(field_type)

            # Handle List types
            origin = get_origin(field_type)
            if origin is list:
                inner_type = get_args(field_type)[0]
                # Handle Optional inner types
                if get_origin(inner_type) is Union:
                    non_none_inner = [
                        arg for arg in get_args(inner_type) if arg is not type(None)
                    ]
                    if len(non_none_inner) == 1:
                        inner_type = non_none_inner[0]

                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    inner_type = self._make_fields_required_and_small(inner_type)
                    field_type = list[inner_type]

            # Set field type and create field
            if self.strict:
                new_annotations[short_name] = field_type
                new_fields[short_name] = Field(
                    ...,
                    alias=field.alias,
                    description=field.description,
                    serialization_alias=name,
                )
            else:
                new_annotations[short_name] = Optional[field_type]
                new_fields[short_name] = Field(
                    default=None,
                    alias=field.alias,
                    description=field.description,
                    serialization_alias=name,
                )

        # Create new class
        class_name = f"Minified{pydantic_cls.__name__}"
        class_dict = {"__annotations__": new_annotations}
        class_dict.update(new_fields)

        return type(class_name, (BaseModel,), class_dict)

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> BaseModel:
        """Parse LLM result and convert back to original schema."""
        parsed = super().parse_result(result, partial=partial)
        return self.get_original(parsed)

    def get_original(self, llm_result: Optional[Union[BaseModel, dict]]) -> BaseModel:
        """Convert minified result back to original schema."""
        if llm_result is None:
            raise ValueError("llm_result cannot be None")

        if isinstance(llm_result, BaseModel):
            data = llm_result.model_dump(by_alias=True)
        else:
            # Clean None values in-place for memory efficiency
            data = self._remove_none_values(llm_result)
            minified = self.minified(**data)
            return self.get_original(minified)

        return self.original_type(**data)

    def _remove_none_values(
        self, data: Union[dict, list, Any]
    ) -> Union[dict, list, Any]:
        """Remove None values to reduce memory usage."""
        if isinstance(data, dict):
            # Use dict comprehension for efficiency
            return {
                k: self._remove_none_values(v) for k, v in data.items() if v is not None
            }
        elif isinstance(data, list):
            # Use list comprehension for efficiency
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data
