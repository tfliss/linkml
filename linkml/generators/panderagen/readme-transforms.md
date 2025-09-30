#### Inline Forms

The following inline forms are supported:

- `inline_dict` - Inlined dictionary representation
    - for nesting single objects
- `inline_list_dict` - Inlined list of dictionaries
    - for inlining multiple objects using a list as the collection
- `inline_collection_dict` - Inlined collection dictionary
    - for inlining multiple objects using a dict as the collection
- `simple_dict` - Simple dictionary form for ranges meeting specific conditions
    - an abbreviated human-readable form of collection dict when there is only one non-id slot.
- `list_foreign_key` - Multivalued foreign key reference
    - list of ids referring to an external table
- `foreign_key` - Single foreign key reference
    - a single id referring to an external table
- `error` - Error state for unsupported configurations

#### Schema Flags

The slot handlers use the following flags from LinkML SchemaView to determine inline forms:

- `multivalued` - Indicates if the slot can contain multiple values
- `inlined` - Indicates if the slot should be inlined rather than referenced
- `inlined_as_list` - Indicates if inlined content should be represented as a list
- `range_has_identifier_or_key` - Whether the slot's range class has an identifier or key
- `range_meets_simple_dict_conditions` - Whether the range meets conditions for simple dictionary representation

#### Backing Form

The simple dict and inline collection dict forms are not efficient for storage and manipulation in many dataframe libraries, so they are mapped to an inlined_list_dict form.