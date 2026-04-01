# Turing Knowledge Graph Entities

| 类别 | 实体 |
|---|---|
| Person | Alan Turing |
| Place | London |
| Organization | University of Cambridge |
| Organization | Princeton University |
| Place | Bletchley Park |
| Organization | University of Manchester |
| Concept | Turing Machine |
| Concept | Turing Test |
| Event | World War II |
| Field | Mathematics |
| Field | Computer Science |
| Field | Cryptography |


```python
triples = [
    ("Alan Turing", "born_in", "London"),
    ("Alan Turing", "educated_at", "University of Cambridge"),
    ("Alan Turing", "educated_at", "Princeton University"),
    ("Alan Turing", "worked_at", "Bletchley Park"),
    ("Alan Turing", "worked_at", "University of Manchester"),
    ("Alan Turing", "proposed", "Turing Machine"),
    ("Alan Turing", "proposed", "Turing Test"),
    ("Alan Turing", "contributed_to", "Cryptography"),
    ("Alan Turing", "contributed_to", "Computer Science"),
    ("Alan Turing", "specialized_in", "Mathematics"),
    ("Bletchley Park", "associated_with", "World War II"),
    ("Turing Machine", "belongs_to", "Computer Science"),
    ("Turing Machine", "related_to", "Mathematics"),
    ("Turing Test", "belongs_to", "Computer Science"),
    ("Cryptography", "associated_with", "World War II")
]
```