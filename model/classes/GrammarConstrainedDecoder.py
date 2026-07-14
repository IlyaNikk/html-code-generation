import json

from .Vocabulary import END_TOKEN


NEWLINE_TOKEN = "\n"
COMMA_TOKEN = ","
OPEN_TOKEN = "{"
CLOSE_TOKEN = "}"


class GrammarConstrainedDecoder:
    def __init__(self, rules_path, vocabulary):
        with open(rules_path) as f:
            rules = json.load(f)

        self.vocabulary = vocabulary
        self.relations = dict(rules.get("relations", {}))
        self.top = rules["top"]
        self.structure = list(rules.get("structure", []))
        self.containers = self._build_containers()
        self.reset()

    def _build_containers(self):
        containers = set([self.top])
        for parent, children in self.relations.items():
            if len(children) > 0:
                containers.add(parent)
        for token in self.structure:
            containers.add(token)
        return containers

    def reset(self):
        self.frames = []
        self.pending_open = None
        self.started = False
        self.closed_top = False
        self.last_type = "start"

    def valid_token_ids(self):
        tokens = self.valid_tokens()
        ids = []
        for token in tokens:
            if token in self.vocabulary.vocabulary:
                ids.append(self.vocabulary.vocabulary[token])
        return ids

    def valid_tokens(self):
        if self.closed_top:
            return [END_TOKEN, NEWLINE_TOKEN]

        if self.pending_open is not None:
            return [OPEN_TOKEN]

        if self.last_type == "start":
            return [self.top]

        if self.last_type == "open":
            return [NEWLINE_TOKEN]

        if self.last_type == "comma":
            return self._valid_children()

        if self.last_type == "newline":
            return [NEWLINE_TOKEN] + self._valid_children() + self._valid_close_tokens()

        if self.last_type in ["leaf", "close"]:
            tokens = [NEWLINE_TOKEN] + self._valid_close_tokens()
            if len(self._valid_children()) > 0 and self.last_type == "leaf":
                tokens.append(COMMA_TOKEN)
            return tokens

        return [NEWLINE_TOKEN] + self._valid_close_tokens()

    def update(self, token):
        if token == NEWLINE_TOKEN:
            self.last_type = "newline"
            return

        if token == COMMA_TOKEN:
            self.last_type = "comma"
            return

        if token == OPEN_TOKEN:
            if self.pending_open is not None:
                self.frames.append({"name": self.pending_open, "children": 0})
                if self.pending_open == self.top:
                    self.started = True
                self.pending_open = None
            self.last_type = "open"
            return

        if token == CLOSE_TOKEN:
            if len(self.frames) > 0:
                closed = self.frames.pop()
                if closed["name"] == self.top:
                    self.closed_top = True
            self.last_type = "close"
            return

        if token == END_TOKEN:
            self.last_type = "end"
            return

        if not self.started and token == self.top:
            self.pending_open = token
            self.last_type = "container"
            return

        if len(self.frames) > 0:
            self.frames[-1]["children"] += 1

        if token in self.containers:
            self.pending_open = token
            self.last_type = "container"
        else:
            self.last_type = "leaf"

    def force_close_tokens(self):
        tokens = []
        if self.pending_open is not None:
            tokens.append(OPEN_TOKEN)
            tokens.append(NEWLINE_TOKEN)
            self.update(OPEN_TOKEN)
            self.update(NEWLINE_TOKEN)

        while len(self.frames) > 0:
            if self.last_type != "newline":
                tokens.append(NEWLINE_TOKEN)
                self.update(NEWLINE_TOKEN)
            tokens.append(CLOSE_TOKEN)
            self.update(CLOSE_TOKEN)

        if self.last_type != "newline":
            tokens.append(NEWLINE_TOKEN)
            self.update(NEWLINE_TOKEN)
        tokens.append(END_TOKEN)
        self.update(END_TOKEN)
        return tokens

    def _valid_children(self):
        if len(self.frames) == 0:
            return []

        frame = self.frames[-1]
        name = frame["name"]
        child_index = frame["children"]

        if name == self.top:
            if child_index < len(self.structure):
                return [self.structure[child_index]]
            return []

        if name == "carousel-wrapper":
            children = self.relations.get(name, [])
            if child_index < len(children):
                return [children[child_index]]
            return []

        return list(self.relations.get(name, []))

    def _valid_close_tokens(self):
        if self._can_close_current_frame():
            return [CLOSE_TOKEN]
        return []

    def _can_close_current_frame(self):
        if len(self.frames) == 0:
            return False

        frame = self.frames[-1]
        name = frame["name"]
        child_count = frame["children"]

        if name == self.top:
            return child_count >= len(self.structure)

        if name == "carousel-wrapper":
            return child_count >= len(self.relations.get(name, []))

        if len(self.relations.get(name, [])) == 0:
            return True

        return child_count > 0
