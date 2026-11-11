"""Evidence-backed extraction schema and source-quote validation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model

SOURCE_FIELDS = ('Title', 'Description', 'Material', 'Fit', 'Color')


def extraction_schema(taxonomy):
    fields = {}
    for name, spec in taxonomy['attributes'].items():
        allowed = Literal[tuple(v for v in spec['values'] if v != 'unknown')]
        record = create_model(
            name.title().replace('_', '') + 'Evidence',
            __config__=ConfigDict(extra='forbid'),
            value=(allowed, ...),
            source_field=(Literal[SOURCE_FIELDS], ...),
            quote=(str, Field(min_length=1)),
        )
        fields[name] = (list[record], Field(default_factory=list,
            description=f"{spec['selection']}: evidence for each value; omit if unsupported. Exact source quotes required."))
    return create_model('EvidenceAttributes', __config__=ConfigDict(extra='forbid'), **fields)


def validate_evidence(attributes, evidence, source, taxonomy):
    """Check complete per-value provenance; does not prove semantic correctness."""
    if set(evidence) - set(taxonomy['attributes']):
        raise ValueError('Evidence contains unknown attributes')
    for name, spec in taxonomy['attributes'].items():
        value = attributes[name]
        values = value if isinstance(value, list) else [value]
        known = set(values) - {'unknown'}
        entries = evidence.get(name, [])
        supported = set()
        for entry in entries:
            if set(entry) != {'value', 'source_field', 'quote'}:
                raise ValueError(f'{name}: evidence requires value, source_field and quote')
            field, quote = entry['source_field'], entry['quote']
            if field not in SOURCE_FIELDS or not isinstance(quote, str) or not quote.strip():
                raise ValueError(f'{name}: invalid source field or blank quote')
            if quote not in str(source.get(field, '')):
                raise ValueError(f'{name}: quote is not an exact substring of {field}')
            if entry['value'] not in known:
                raise ValueError(f'{name}: evidence does not match extracted values')
            supported.add(entry['value'])
        if supported != known:
            raise ValueError(f'{name}: every known value requires evidence')
        if len(values) != len(set(values)):
            raise ValueError(f'{name}: duplicate values')
        if spec['selection'] == 'multi' and len(values) > 1 and set(values) & {'none', 'unknown', 'not_applicable'}:
            raise ValueError(f'{name}: exclusive sentinel combined with other values')


def unpack_evidence(result, source, taxonomy):
    attributes, evidence = {}, {}
    for name, spec in taxonomy['attributes'].items():
        entries = result.model_dump()[name]
        values = list(dict.fromkeys(e['value'] for e in entries))
        if spec['selection'] == 'single':
            if len(values) > 1:
                raise ValueError(f'{name}: single attribute has conflicting values')
            attributes[name] = values[0] if values else 'unknown'
        else:
            attributes[name] = values or ['unknown']
        if entries:
            evidence[name] = entries
    validate_evidence(attributes, evidence, source, taxonomy)
    return attributes, evidence


def sanitize_evidence(result, source, taxonomy):
    """Discard invalid quote claims without weakening validation of saved values.

    Preserve rejected claims in warnings for review. Do not relocate, normalize,
    or fabricate quotes. Conflicting values become unknown instead of guessing.
    """
    data = result.model_dump()
    warnings = []
    for name, spec in taxonomy['attributes'].items():
        accepted = []
        for entry in data[name]:
            quote = entry['quote']
            if not quote.strip() or quote not in str(source.get(entry['source_field'], '')):
                warnings.append({'attribute': name, 'rejected_evidence': entry,
                                 'reason': 'Quote is not an exact nonblank substring of the named source field'})
            else:
                accepted.append(entry)
        values = {entry['value'] for entry in accepted}
        conflict = ((spec['selection'] == 'single' and len(values) > 1) or
                    (spec['selection'] == 'multi' and len(values) > 1 and
                     bool(values & {'none', 'unknown', 'not_applicable'})))
        if conflict:
            warnings.append({'attribute': name, 'rejected_evidence': accepted,
                             'reason': 'Conflicting values; attribute left unknown'})
            accepted = []
        data[name] = accepted
    clean = type(result).model_validate(data)
    attributes, evidence = unpack_evidence(clean, source, taxonomy)
    return attributes, evidence, warnings


def parse_evidence_response(response, schema, source, taxonomy):
    """Recover valid individual claims from a schema-invalid tool response.

    Keep the strict enum schema sent to the model. Invalid claims are rejected
    locally and recorded, never moved to another attribute or accepted as facts.
    """
    from pydantic import TypeAdapter, ValidationError
    parsed = response.get('parsed')
    if parsed is not None:
        return sanitize_evidence(parsed, source, taxonomy)
    calls = getattr(response.get('raw'), 'tool_calls', [])
    if len(calls) != 1 or not isinstance(calls[0].get('args'), dict):
        raise ValueError('Expected one structured evidence tool response')
    raw = calls[0]['args']
    clean, rejected = {}, []
    for name, entries in raw.items():
        if name not in schema.model_fields or not isinstance(entries, list):
            rejected.append({'attribute':name, 'rejected_evidence':entries,
                             'reason':'Unknown attribute or non-list evidence'})
            continue
        entry_type = schema.model_fields[name].annotation.__args__[0]
        validator = TypeAdapter(entry_type)
        clean[name] = []
        for entry in entries:
            try:
                clean[name].append(validator.validate_python(entry).model_dump())
            except ValidationError:
                rejected.append({'attribute':name, 'rejected_evidence':entry,
                                 'reason':'Value, source field, or evidence structure violates schema'})
    if not clean:
        raise ValueError('No recognizable attribute lists in model response')
    attributes, evidence, warnings = sanitize_evidence(schema.model_validate(clean), source, taxonomy)
    return attributes, evidence, rejected + warnings
