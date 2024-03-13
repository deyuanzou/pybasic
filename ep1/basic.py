#########################
# Token 词法单元
#########################
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_UNDEFINED = 'UNDEFINED'


class Token:
    def __init__(self, type_, value):  # 这里使用type_是为了防止将内置名称type覆盖
        self.type = type_
        self.value = value

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'


#########################
#  Lexer 词法分析器
#########################
class Lexer:
    def __init__(self, text):
        self.text = text  # text为输入文本
        self.pos = -1  # pos为当前字符的位置
        self.current_char = None  # current_char为当前字符

    def read_a_char(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.read_a_char()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, '+'))
                self.read_a_char()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, '-'))
                self.read_a_char()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, '*'))
                self.read_a_char()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, '/'))
                self.read_a_char()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, '('))
                self.read_a_char()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, ')'))
                self.read_a_char()
            else:
                tokens.append(Token(TT_UNDEFINED, 'undefined'))
