#########################
# Specialness 作特殊处理的值
#########################
DIGITS = '0123456789'


#########################
# Error 错误
#########################
class Error:
    def __init__(self, error_name, details):
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}:{self.details}'
        return result


class IllegalCharError(Error):
    def __init__(self, details):
        super().__init__('IllegalCharError', details)


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


class Token:
    def __init__(self, type_, value=None):  # 这里使用type_是为了防止将内置名称type覆盖
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
        self.read_a_char()

    def read_a_char(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.read_a_char()
            elif self.current_char in DIGITS:
                tokens.append(self.make_num())
                self.read_a_char()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.read_a_char()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.read_a_char()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.read_a_char()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.read_a_char()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.read_a_char()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.read_a_char()
            else:
                char = self.current_char
                self.read_a_char()
                return [], IllegalCharError("'" + char + "'")

        return tokens, None

    def make_num(self):
        num_str = ''
        dot_count = 0

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.read_a_char()

        # 在make_num()的while循环作判断前向前读取了一个字符，
        # 回到make_tokens()的while循环时又向前读取了一个字符，
        # 因此需要回退一个字符。
        self.pos -= 1

        if dot_count == 0:
            return Token(TT_INT, int(num_str))
        else:
            return Token(TT_FLOAT, float(num_str))


#########################
# Run 主程序
#########################
def run(text):
    lexer = Lexer(text)
    tokens, error = lexer.make_tokens()

    return tokens, error
