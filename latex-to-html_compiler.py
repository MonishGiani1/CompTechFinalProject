import re
from typing import List, Tuple, Optional
from enum import Enum, auto

# ============== LEXER ==============

class TokenType(Enum):
    # Document structure
    BEGIN = auto()
    END = auto()
    SECTION = auto()
    SUBSECTION = auto()
    
    # Text formatting
    TEXTBF = auto()      
    TEXTIT = auto()      
    UNDERLINE = auto()   
    
    # Lists
    ITEMIZE = auto()
    ENUMERATE = auto()
    ITEM = auto()
    
    # Special characters
    LBRACE = auto()      
    RBRACE = auto()      
    BACKSLASH = auto()   
    
    # Content
    TEXT = auto()
    NEWLINE = auto()
    EOF = auto()

class Token:
    def __init__(self, type: TokenType, value: str, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.col})"

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens = []
    
    def error(self, msg: str):
        raise SyntaxError(f"Lexer error at {self.line}:{self.col}: {msg}")
    
    def peek(self, offset=0) -> Optional[str]:
        pos = self.pos + offset
        return self.text[pos] if pos < len(self.text) else None
    
    def advance(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        char = self.text[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return char
    
    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t':
            self.advance()
    
    def read_command(self) -> str:
        """Read a LaTeX command like \\section"""
        cmd = ""
        while self.peek() and self.peek().isalpha():
            cmd += self.advance()
        return cmd
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            start_line, start_col = self.line, self.col
            char = self.peek()
            
            if char == '\\':
                self.advance()  # consume backslash
                cmd = self.read_command()
                
                # Map commands to token types
                cmd_map = {
                    'begin': TokenType.BEGIN,
                    'end': TokenType.END,
                    'section': TokenType.SECTION,
                    'subsection': TokenType.SUBSECTION,
                    'textbf': TokenType.TEXTBF,
                    'textit': TokenType.TEXTIT,
                    'underline': TokenType.UNDERLINE,
                    'item': TokenType.ITEM,
                }
                
                if cmd in cmd_map:
                    self.tokens.append(Token(cmd_map[cmd], cmd, start_line, start_col))
                else:
                    # Unknown command, treat as text
                    self.tokens.append(Token(TokenType.TEXT, '\\' + cmd, start_line, start_col))
            
            elif char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', start_line, start_col))
            
            elif char == '}':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, '}', start_line, start_col))
            
            elif char == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', start_line, start_col))
            
            elif char in ' \t':
                self.skip_whitespace()
            
            else:
                # Read text until special character
                text = ""
                while self.peek() and self.peek() not in '\\{}\n':
                    text += self.advance()
                if text:
                    self.tokens.append(Token(TokenType.TEXT, text, start_line, start_col))
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return self.tokens

# ============== PARSER ==============

class ASTNode:
    pass

class Document(ASTNode):
    def __init__(self, children: List[ASTNode]):
        self.children = children

class Environment(ASTNode):
    def __init__(self, name: str, children: List[ASTNode]):
        self.name = name
        self.children = children

class Command(ASTNode):
    def __init__(self, name: str, content: ASTNode):
        self.name = name
        self.content = content

class Text(ASTNode):
    def __init__(self, value: str):
        self.value = value

class Section(ASTNode):
    def __init__(self, title: str, level: int = 1):
        self.title = title
        self.level = level

class ListItem(ASTNode):
    def __init__(self, content: List[ASTNode]):
        self.content = content

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def error(self, msg: str):
        token = self.current()
        raise SyntaxError(f"Parser error at {token.line}:{token.col}: {msg}")
    
    def current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]
    
    def peek(self, offset=0) -> Token:
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]
    
    def advance(self) -> Token:
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current()
        if token.type != token_type:
            self.error(f"Expected {token_type.name}, got {token.type.name}")
        return self.advance()
    
    def skip_newlines(self):
        while self.current().type == TokenType.NEWLINE:
            self.advance()
    
    def parse(self) -> Document:
        children = []
        while self.current().type != TokenType.EOF:
            self.skip_newlines()
            if self.current().type == TokenType.EOF:
                break
            children.append(self.parse_statement())
        return Document(children)
    
    def parse_statement(self) -> ASTNode:
        token = self.current()
        
        if token.type == TokenType.BEGIN:
            return self.parse_environment()
        elif token.type in [TokenType.SECTION, TokenType.SUBSECTION]:
            return self.parse_section()
        elif token.type in [TokenType.TEXTBF, TokenType.TEXTIT, TokenType.UNDERLINE]:
            return self.parse_command()
        elif token.type == TokenType.ITEM:
            return self.parse_item()
        elif token.type == TokenType.TEXT:
            return Text(self.advance().value)
        elif token.type == TokenType.NEWLINE:
            self.advance()
            return Text('\n')
        else:
            self.error(f"Unexpected token: {token.type.name}")
    
    def parse_environment(self) -> Environment:
        self.expect(TokenType.BEGIN)
        self.expect(TokenType.LBRACE)
        name = self.expect(TokenType.TEXT).value
        self.expect(TokenType.RBRACE)
        
        children = []
        while not (self.current().type == TokenType.END):
            self.skip_newlines()
            if self.current().type == TokenType.END:
                break
            children.append(self.parse_statement())
        
        self.expect(TokenType.END)
        self.expect(TokenType.LBRACE)
        self.expect(TokenType.TEXT)
        self.expect(TokenType.RBRACE)
        
        return Environment(name, children)
    
    def parse_section(self) -> Section:
        token = self.advance()
        level = 1 if token.type == TokenType.SECTION else 2
        self.expect(TokenType.LBRACE)
        title = self.expect(TokenType.TEXT).value
        self.expect(TokenType.RBRACE)
        return Section(title, level)
    
    def parse_command(self) -> Command:
        token = self.advance()
        self.expect(TokenType.LBRACE)
        content = self.parse_content()
        self.expect(TokenType.RBRACE)
        return Command(token.value, content)
    
    def parse_content(self) -> ASTNode:
        """Parse content inside braces"""
        parts = []
        while self.current().type not in [TokenType.RBRACE, TokenType.EOF]:
            if self.current().type == TokenType.TEXT:
                parts.append(Text(self.advance().value))
            elif self.current().type in [TokenType.TEXTBF, TokenType.TEXTIT, TokenType.UNDERLINE]:
                parts.append(self.parse_command())
            else:
                break
        
        if len(parts) == 1:
            return parts[0]
        return Document(parts)
    
    def parse_item(self) -> ListItem:
        self.expect(TokenType.ITEM)
        content = []
        while self.current().type not in [TokenType.ITEM, TokenType.END, TokenType.EOF, TokenType.NEWLINE]:
            if self.current().type == TokenType.TEXT:
                content.append(Text(self.advance().value))
            elif self.current().type in [TokenType.TEXTBF, TokenType.TEXTIT]:
                content.append(self.parse_command())
        return ListItem(content)

# ============== CODE GENERATOR ==============

class HTMLGenerator:
    def __init__(self):
        self.html = []
    
    def generate(self, node: ASTNode) -> str:
        self.html = []
        self.visit(node)
        return ''.join(self.html)
    
    def visit(self, node: ASTNode):
        if isinstance(node, Document):
            self.visit_document(node)
        elif isinstance(node, Environment):
            self.visit_environment(node)
        elif isinstance(node, Command):
            self.visit_command(node)
        elif isinstance(node, Text):
            self.visit_text(node)
        elif isinstance(node, Section):
            self.visit_section(node)
        elif isinstance(node, ListItem):
            self.visit_list_item(node)
    
    def visit_document(self, node: Document):
        self.html.append('<!DOCTYPE html>\n<html>\n<head>\n')
        self.html.append('<meta charset="UTF-8">\n')
        self.html.append('<title>LaTeX Document</title>\n')
        self.html.append('</head>\n<body>\n')
        for child in node.children:
            self.visit(child)
        self.html.append('</body>\n</html>')
    
    def visit_environment(self, node: Environment):
        if node.name == 'itemize':
            self.html.append('<ul>\n')
            for child in node.children:
                self.visit(child)
            self.html.append('</ul>\n')
        elif node.name == 'enumerate':
            self.html.append('<ol>\n')
            for child in node.children:
                self.visit(child)
            self.html.append('</ol>\n')
        elif node.name == 'document':
            for child in node.children:
                self.visit(child)
        else:
            self.html.append(f'<div class="{node.name}">\n')
            for child in node.children:
                self.visit(child)
            self.html.append('</div>\n')
    
    def visit_command(self, node: Command):
        tag_map = {
            'textbf': 'strong',
            'textit': 'em',
            'underline': 'u',
        }
        tag = tag_map.get(node.name, 'span')
        self.html.append(f'<{tag}>')
        self.visit(node.content)
        self.html.append(f'</{tag}>')
    
    def visit_text(self, node: Text):
        self.html.append(node.value)
    
    def visit_section(self, node: Section):
        level = node.level
        self.html.append(f'<h{level}>{node.title}</h{level}>\n')
    
    def visit_list_item(self, node: ListItem):
        self.html.append('<li>')
        for item in node.content:
            self.visit(item)
        self.html.append('</li>\n')

# ============== MAIN COMPILER ==============

def compile_latex_to_html(latex_code: str) -> str:
    """Main compilation function"""
    try:
        # Lexical analysis
        lexer = Lexer(latex_code)
        tokens = lexer.tokenize()
        
        # Parsing
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Code generation
        generator = HTMLGenerator()
        html = generator.generate(ast)
        
        return html
    except SyntaxError as e:
        return f"Error: {e}"

# ============== TESTING ==============

if __name__ == "__main__":
    latex_sample = r"""
\begin{document}
\section{Introduction}
This is a \textbf{bold} statement and this is \textit{italic}.

\subsection{Features}
Here is a list:
\begin{itemize}
\item First item
\item Second item with \textbf{bold text}
\end{itemize}

\begin{enumerate}
\item Numbered one
\item Numbered two
\end{enumerate}
\end{document}
"""
    
    html_output = compile_latex_to_html(latex_sample)
    print(html_output)
    
 