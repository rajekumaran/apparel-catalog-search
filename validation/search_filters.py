"""Validate parsed shopper intent with Pydantic and build safe DuckDB SQL.

An LLM can return the SearchFilters JSON shape; validation also runs locally.
No API calls are made by this module.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
TAXONOMIES = {
    category: json.loads((ROOT / f'data/womens_{category.lower()}_taxonomy.json').read_text())
    for category in ('Tops', 'Bottoms', 'Dresses')
}


class AttributeFilter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    # OR within a filter; AND across filters.
    values: list[str] = Field(min_length=1)


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    category: Literal['Tops', 'Bottoms', 'Dresses']
    currency: Literal['USD'] = 'USD'  # Explicit store default for this prototype.
    price_under: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    attributes: list[AttributeFilter] = Field(default_factory=list)
    semantic_query: str | None = None  # Ranking text, never interpreted as SQL.

    @model_validator(mode='after')
    def check_taxonomy(self):
        if self.category is None:
            if self.attributes:
                raise ValueError('Attribute filters require a category')
            return self
        if self.category not in TAXONOMIES:
            raise ValueError('Unsupported category')
        specs = TAXONOMIES[self.category]['attributes']
        names = set()
        for item in self.attributes:
            if item.name in names:
                raise ValueError(f'Duplicate attribute filter: {item.name}')
            names.add(item.name)
            if item.name not in specs:
                raise ValueError(f'{item.name} is not an attribute for {self.category}')
            if len(set(item.values)) != len(item.values):
                raise ValueError(f'Duplicate values for {item.name}')
            for value in item.values:
                if value == 'unknown' or value not in specs[item.name]['values']:
                    raise ValueError(f'Invalid search value {item.name}={value}')
        return self


def build_search_sql(filters: SearchFilters, limit: int = 20):
    """Select the latest imported result per product in the current taxonomy.

    Unknown attributes do not satisfy requirements. All user values are bound
    parameters, including attribute names. Multi fields match any requested value.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError('limit must be an integer from 1 to 100')
    taxonomy = TAXONOMIES[filters.category]
    sql = '''WITH current_extractions AS (
        SELECT * FROM extractions
        WHERE taxonomy_category = ? AND taxonomy_version = ? AND status = 'completed'
        QUALIFY row_number() OVER (
            PARTITION BY product_id
            ORDER BY extracted_at DESC NULLS LAST, extraction_id DESC
        ) = 1
    )
    SELECT p.product_id, p.title, p.price, p.currency, e.extraction_id
    FROM products p JOIN current_extractions e USING (product_id)
    WHERE p.category = ? AND p.currency = ?'''
    params = [taxonomy['category'], taxonomy['version'], filters.category, filters.currency]
    if filters.price_under is not None:
        sql += '\n AND p.price < ?'
        params.append(filters.price_under)
    for item in filters.attributes:
        placeholders = ', '.join('?' for _ in item.values)
        sql += f'''\n AND EXISTS (
            SELECT 1 FROM product_attributes a
            WHERE a.extraction_id = e.extraction_id
              AND a.attribute_name = ? AND a.attribute_value IN ({placeholders})
        )'''
        params.extend([item.name, *item.values])
    sql += '\n ORDER BY p.price, p.product_id LIMIT ?'
    params.append(limit)
    return sql, params


if __name__ == '__main__':
    import duckdb

    # Example structured output for: "blue high-rise jeans under $80".
    filters = SearchFilters.model_validate({
        'category': 'Bottoms',
        'price_under': 80,
        'attributes': [
            {'name': 'bottom_type', 'values': ['jeans']},
            {'name': 'color_primary', 'values': ['blue']},
            {'name': 'rise', 'values': ['high']},
        ],
    })
    sql, params = build_search_sql(filters)
    with duckdb.connect(str(ROOT / 'data/catalog_attributes.duckdb'), read_only=True) as con:
        print(con.execute(sql, params).fetchdf().to_string(index=False))
