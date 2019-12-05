from enum import Enum

class RelationType(Enum):
    PROGRESSION = 1
    XOR = 2
    OR = 3
    AND = 4
    CONSISTENT_UNION = 5

class ObjectType(Enum):
    SHAPE = 1
    LINE = 2

class AttributeType(Enum):
    SIZE = 1
    TYPE = 2
    COLOR = 3
    POSITION = 4
    NUMBER = 5

class Triple:
    def __init__(self, relation_type, object_type, attribute_type):
        self.relation_type = relation_type
        self.object_type = object_type
        self.attribute_type = attribute_type

    def __eq__(self, other):
        return self.relation_type == other.relation_type and \
            self.object_type == other.object_type and \
            self.attribute_type == other.attribute_type

    def __hash__(self):
        return hash('{0}_{1}_{2}'.format(
            self.relation_type,
            self.object_type,
            self.attribute_type))

    def to_str(self):
        return '(' + str(self.relation_type.name) + ',' + str(self.object_type.name) + ',' + str(
            self.attribute_type.name) + ')'

class Structure:
    def __init__(self):
        self.triples = set()

    def add(self, triple):
        assert(triple not in self.triples)
        self.triples.add(triple)

    def __eq__(self, other):
        return self.triples == other.triples

    def to_str(self):
        triple_strs = []
        for triple in self.triples:
            triple_strs.append(triple.to_str())
        return ','.join(triple_strs)