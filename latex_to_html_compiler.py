import streamlit as st
import re
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto
import base64


# ============== SYMBOL TABLE ==============

class Symbol:
    """Represents a symbol in the symbol table"""
    def __init__(self, name: str, symbol_type: str, value: Any, line: int, col: int):
        self.name = name
        self.symbol_type = symbol_type  # 'label', 'macro', 'counter'
        self.value = value
        self.line = line
        self.col = col
        self.references = []  # List of (line, col) where this symbol is referenced

class SymbolTable:
    """Symbol table for tracking labels, macros, and counters"""
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.counters = {
            'section': 0,
            'subsection': 0,
            'subsubsection': 0,
            'figure': 0,
            'table': 0,
            'equation': 0
        }
        self.current_section_number = ""
        self.warnings = []
        
    def add_symbol(self, name: str, symbol_type: str, value: Any, line: int, col: int):
        """Add a symbol to the table"""
        if name in self.symbols:
            self.warnings.append(f"Warning at {line}:{col}: Symbol '{name}' redefined")
        self.symbols[name] = Symbol(name, symbol_type, value, line, col)
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol"""
        return self.symbols.get(name)
    
    def add_reference(self, name: str, line: int, col: int):
        """Add a reference to a symbol"""
        symbol = self.lookup(name)
        if symbol:
            symbol.references.append((line, col))
        else:
            self.warnings.append(f"Warning at {line}:{col}: Undefined reference '{name}'")
    
    def increment_counter(self, counter_type: str) -> int:
        """Increment and return a counter"""
        if counter_type in self.counters:
            self.counters[counter_type] += 1
            
            # Update section numbering
            if counter_type == 'section':
                self.current_section_number = str(self.counters['section'])
                self.counters['subsection'] = 0
                self.counters['subsubsection'] = 0
            elif counter_type == 'subsection':
                self.current_section_number = f"{self.counters['section']}.{self.counters['subsection']}"
                self.counters['subsubsection'] = 0
            elif counter_type == 'subsubsection':
                self.current_section_number = f"{self.counters['section']}.{self.counters['subsection']}.{self.counters['subsubsection']}"
            
            return self.counters[counter_type]
        return 0
    
    def get_counter(self, counter_type: str) -> int:
        """Get current counter value"""
        return self.counters.get(counter_type, 0)
    
    def get_section_number(self) -> str:
        """Get current section number"""
        return self.current_section_number


# ============== MACRO PROCESSOR ==============

class Macro:
    """Represents a user-defined macro"""
    def __init__(self, name: str, num_params: int, definition: str):
        self.name = name
        self.num_params = num_params
        self.definition = definition

class MacroProcessor:
    """Handles macro definitions and expansion"""
    def __init__(self):
        self.macros: Dict[str, Macro] = {}
        # Built-in macros
        self.add_builtin_macros()
    
    def add_builtin_macros(self):
        """Add common LaTeX shortcuts"""
        builtins = {
            'LaTeX': r'\textsc{La}\TeX',
            'TeX': r'\textsc{TeX}',
        }
        for name, definition in builtins.items():
            self.macros[name] = Macro(name, 0, definition)
    
    def define_macro(self, name: str, num_params: int, definition: str):
        """Define a new macro"""
        self.macros[name] = Macro(name, num_params, definition)
    
    def has_macro(self, name: str) -> bool:
        """Check if a macro is defined"""
        return name in self.macros
    
    def expand_macro(self, name: str, params: List[str]) -> str:
        """Expand a macro with given parameters"""
        if name not in self.macros:
            return f"\\{name}"
        
        macro = self.macros[name]
        if len(params) != macro.num_params:
            return f"\\{name}"  # Parameter mismatch, return as-is
        
        # Perform parameter substitution
        result = macro.definition
        for i, param in enumerate(params, 1):
            result = result.replace(f"#{i}", param)
        
        return result


# ============== LEXER ==============

class TokenType(Enum):
    # Document structure
    BEGIN = auto()
    END = auto()
    SECTION = auto()
    SUBSECTION = auto()
    SUBSUBSECTION = auto()
    PARAGRAPH = auto()

    # Text formatting
    TEXTBF = auto()
    TEXTIT = auto()
    UNDERLINE = auto()
    EMPH = auto()
    TEXTTT = auto()
    TEXTRM = auto()
    TEXTSC = auto()

    # Font sizes
    TINY = auto()
    SMALL = auto()
    LARGE = auto()
    HUGE = auto()

    # Lists
    ITEMIZE = auto()
    ENUMERATE = auto()
    DESCRIPTION = auto()
    ITEM = auto()

    # Math
    MATH_INLINE = auto()
    MATH_DISPLAY = auto()
    MATH_COMMAND = auto()

    # Links and references
    HREF = auto()
    URL = auto()
    LABEL = auto()
    REF = auto()

    # Macros
    NEWCOMMAND = auto()
    RENEWCOMMAND = auto()

    # Images and figures
    INCLUDEGRAPHICS = auto()
    FIGURE = auto()
    CAPTION = auto()
    CENTERING = auto()

    # Tables
    TABULAR = auto()
    HLINE = auto()
    TABLE = auto()

    # Code
    VERBATIM = auto()
    LSTLISTING = auto()

    # Special formatting
    QUOTE = auto()
    QUOTATION = auto()
    VERSE = auto()
    CENTER = auto()
    FLUSHLEFT = auto()
    FLUSHRIGHT = auto()

    # Text color
    TEXTCOLOR = auto()
    COLOR = auto()

    # Special characters
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    BACKSLASH = auto()
    AMPERSAND = auto()
    DOLLAR = auto()
    CARET = auto()
    UNDERSCORE = auto()

    # Content
    TEXT = auto()
    NEWLINE = auto()
    COMMENT = auto()
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
        """Read a LaTeX command"""
        cmd = ""
        while self.peek() and self.peek().isalpha():
            cmd += self.advance()
        return cmd

    def read_comment(self) -> str:
        """Read a comment until end of line"""
        comment = ""
        while self.peek() and self.peek() != '\n':
            comment += self.advance()
        return comment

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            start_line, start_col = self.line, self.col
            char = self.peek()

            # Handle comments
            if char == '%':
                self.advance()
                comment = self.read_comment()
                self.tokens.append(Token(TokenType.COMMENT, comment, start_line, start_col))
                continue

            # Handle math mode
            if char == '$':
                self.advance()
                if self.peek() == '$':
                    self.advance()
                    self.tokens.append(Token(TokenType.MATH_DISPLAY, '$$', start_line, start_col))
                else:
                    self.tokens.append(Token(TokenType.MATH_INLINE, '$', start_line, start_col))
                continue

            if char == '\\':
                self.advance()
                next_char = self.peek()

                # Handle special backslash commands
                if next_char == '[':
                    self.advance()
                    self.tokens.append(Token(TokenType.MATH_DISPLAY, '\\[', start_line, start_col))
                    continue
                elif next_char == ']':
                    self.advance()
                    self.tokens.append(Token(TokenType.MATH_DISPLAY, '\\]', start_line, start_col))
                    continue
                elif next_char == '(':
                    self.advance()
                    self.tokens.append(Token(TokenType.MATH_INLINE, '\\(', start_line, start_col))
                    continue
                elif next_char == ')':
                    self.advance()
                    self.tokens.append(Token(TokenType.MATH_INLINE, '\\)', start_line, start_col))
                    continue
                elif next_char == '\\':
                    self.advance()
                    self.tokens.append(Token(TokenType.NEWLINE, '\\\\', start_line, start_col))
                    continue

                cmd = self.read_command()

                # Map commands to token types
                cmd_map = {
                    'begin': TokenType.BEGIN,
                    'end': TokenType.END,
                    'section': TokenType.SECTION,
                    'subsection': TokenType.SUBSECTION,
                    'subsubsection': TokenType.SUBSUBSECTION,
                    'paragraph': TokenType.PARAGRAPH,
                    'textbf': TokenType.TEXTBF,
                    'textit': TokenType.TEXTIT,
                    'underline': TokenType.UNDERLINE,
                    'emph': TokenType.EMPH,
                    'texttt': TokenType.TEXTTT,
                    'textrm': TokenType.TEXTRM,
                    'textsc': TokenType.TEXTSC,
                    'tiny': TokenType.TINY,
                    'small': TokenType.SMALL,
                    'large': TokenType.LARGE,
                    'Large': TokenType.LARGE,
                    'huge': TokenType.HUGE,
                    'Huge': TokenType.HUGE,
                    'item': TokenType.ITEM,
                    'href': TokenType.HREF,
                    'url': TokenType.URL,
                    'label': TokenType.LABEL,
                    'ref': TokenType.REF,
                    'newcommand': TokenType.NEWCOMMAND,
                    'renewcommand': TokenType.RENEWCOMMAND,
                    'includegraphics': TokenType.INCLUDEGRAPHICS,
                    'caption': TokenType.CAPTION,
                    'centering': TokenType.CENTERING,
                    'hline': TokenType.HLINE,
                    'textcolor': TokenType.TEXTCOLOR,
                    'color': TokenType.COLOR,
                }

                # Known math commands
                known_math_commands = {
                    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
                    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
                    'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
                    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
                    'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
                    'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
                    'frac', 'sqrt', 'sum', 'prod', 'int', 'lim', 'infty',
                    'sin', 'cos', 'tan', 'log', 'ln', 'exp',
                    'leq', 'geq', 'neq', 'approx', 'equiv', 'times', 'div',
                    'partial', 'nabla', 'pm', 'mp', 'cdot', 'ldots', 'cdots',
                }

                if cmd in cmd_map:
                    self.tokens.append(Token(cmd_map[cmd], cmd, start_line, start_col))
                elif cmd in known_math_commands:
                    self.tokens.append(Token(TokenType.MATH_COMMAND, cmd, start_line, start_col))
                elif cmd:
                    # Unknown command - treat as text (could be user-defined macro)
                    self.tokens.append(Token(TokenType.TEXT, '\\' + cmd, start_line, start_col))
                continue

            # Handle special characters
            if char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', start_line, start_col))
                continue
            if char == '}':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, '}', start_line, start_col))
                continue
            if char == '[':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', start_line, start_col))
                continue
            if char == ']':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACKET, ']', start_line, start_col))
                continue
            if char == '&':
                self.advance()
                self.tokens.append(Token(TokenType.AMPERSAND, '&', start_line, start_col))
                continue
            if char == '\n':
                self.advance()
                if self.peek() == '\n':
                    while self.peek() == '\n':
                        self.advance()
                    self.tokens.append(Token(TokenType.NEWLINE, '\n\n', start_line, start_col))
                else:
                    self.tokens.append(Token(TokenType.NEWLINE, '\n', start_line, start_col))
                continue

            # Regular text
            text = ""
            while (self.peek() and self.peek() not in '\\{}[]$%&\n'):
                text += self.advance()
            if text:
                self.tokens.append(Token(TokenType.TEXT, text, start_line, start_col))

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return self.tokens


# ============== AST NODES ==============

class ASTNode:
    pass


class Document(ASTNode):
    def __init__(self, content: List[ASTNode]):
        self.content = content


class Section(ASTNode):
    def __init__(self, level: int, title: str, content: List[ASTNode], number: str = ""):
        self.level = level
        self.title = title
        self.content = content
        self.number = number  # Section number like "1.2.3"


class TextNode(ASTNode):
    def __init__(self, text: str):
        self.text = text


class FormattedText(ASTNode):
    def __init__(self, format_type: str, content: List[ASTNode]):
        self.format_type = format_type
        self.content = content


class MathNode(ASTNode):
    def __init__(self, content: str, is_display: bool = False, number: int = 0):
        self.content = content
        self.is_display = is_display
        self.number = number  # Equation number


class ListNode(ASTNode):
    def __init__(self, list_type: str, items: List[List[ASTNode]]):
        self.list_type = list_type
        self.items = items


class TableNode(ASTNode):
    def __init__(self, alignment: str, rows: List[List[List[ASTNode]]], number: int = 0):
        self.alignment = alignment
        self.rows = rows
        self.number = number


class LinkNode(ASTNode):
    def __init__(self, url: str, text: List[ASTNode]):
        self.url = url
        self.text = text


class ImageNode(ASTNode):
    def __init__(self, path: str, options: Dict[str, str] = None):
        self.path = path
        self.options = options or {}


class EnvironmentNode(ASTNode):
    def __init__(self, env_type: str, content: List[ASTNode]):
        self.env_type = env_type
        self.content = content


class LabelNode(ASTNode):
    def __init__(self, name: str):
        self.name = name


class RefNode(ASTNode):
    def __init__(self, name: str):
        self.name = name


class MacroDefNode(ASTNode):
    def __init__(self, name: str, num_params: int, definition: str):
        self.name = name
        self.num_params = num_params
        self.definition = definition


# ============== PARSER ==============

class Parser:
    def __init__(self, tokens: List[Token], macro_processor: MacroProcessor, symbol_table: SymbolTable):
        self.tokens = tokens
        self.pos = 0
        self.macro_processor = macro_processor
        self.symbol_table = symbol_table

    def error(self, msg: str):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            raise SyntaxError(f"Parser error at {token.line}:{token.col}: {msg}")
        raise SyntaxError(f"Parser error at end of input: {msg}")

    def peek(self, offset=0) -> Token:
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        token = self.peek()
        if token.type != token_type:
            self.error(f"Expected {token_type.name}, got {token.type.name}")
        return self.advance()

    def skip_comments_and_newlines(self):
        while self.peek().type in [TokenType.COMMENT, TokenType.NEWLINE]:
            self.advance()

    def read_braced_content(self) -> List[ASTNode]:
        """Read content inside braces"""
        if self.peek().type != TokenType.LBRACE:
            return []
        
        self.expect(TokenType.LBRACE)
        content = []
        depth = 1

        while depth > 0 and self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.LBRACE:
                depth += 1
                self.advance()
            elif self.peek().type == TokenType.RBRACE:
                depth -= 1
                if depth > 0:
                    self.advance()
            else:
                content.extend(self.parse_inline())

        if self.peek().type == TokenType.RBRACE:
            self.expect(TokenType.RBRACE)
        return content

    def read_braced_text(self) -> str:
        """Read text inside braces as a string"""
        if self.peek().type != TokenType.LBRACE:
            return ""
        
        self.expect(TokenType.LBRACE)
        text = ""
        depth = 1

        while depth > 0 and self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.LBRACE:
                depth += 1
                text += "{"
                self.advance()
            elif self.peek().type == TokenType.RBRACE:
                depth -= 1
                if depth > 0:
                    text += "}"
                    self.advance()
            else:
                text += self.peek().value
                self.advance()

        if self.peek().type == TokenType.RBRACE:
            self.expect(TokenType.RBRACE)
        return text
    
    def read_braced_text_preserving_commands(self) -> str:
        """Read text inside braces as a string, preserving LaTeX commands with backslashes"""
        if self.peek().type != TokenType.LBRACE:
            return ""
        
        self.expect(TokenType.LBRACE)
        text = ""
        depth = 1
        
        # Map token types to their LaTeX command names
        command_names = {
            TokenType.TEXTBF: 'textbf',
            TokenType.TEXTIT: 'textit',
            TokenType.TEXTCOLOR: 'textcolor',
            TokenType.UNDERLINE: 'underline',
            TokenType.EMPH: 'emph',
            TokenType.TEXTTT: 'texttt',
            TokenType.TEXTRM: 'textrm',
            TokenType.TEXTSC: 'textsc',
            TokenType.TINY: 'tiny',
            TokenType.SMALL: 'small',
            TokenType.LARGE: 'large',
            TokenType.HUGE: 'huge',
        }

        while depth > 0 and self.peek().type != TokenType.EOF:
            token = self.peek()
            
            if token.type == TokenType.LBRACE:
                depth += 1
                text += "{"
                self.advance()
            elif token.type == TokenType.RBRACE:
                depth -= 1
                if depth > 0:
                    text += "}"
                    self.advance()
            elif token.type in command_names:
                # Reconstruct the command with backslash
                text += "\\" + command_names[token.type]
                self.advance()
            else:
                text += token.value
                self.advance()

        if self.peek().type == TokenType.RBRACE:
            self.expect(TokenType.RBRACE)
        return text

    def read_optional_arg(self) -> Optional[str]:
        """Read optional argument in brackets"""
        if self.peek().type == TokenType.LBRACKET:
            self.advance()
            arg = ""
            while self.peek().type != TokenType.RBRACKET and self.peek().type != TokenType.EOF:
                arg += self.peek().value
                self.advance()
            self.expect(TokenType.RBRACKET)
            return arg
        return None

    def parse(self) -> Document:
        """Parse the token stream into an AST"""
        content = []

        # Skip document preamble if it exists
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.BEGIN and self.peek(1).type == TokenType.LBRACE:
                self.advance()
                self.advance()
                if self.peek().type == TokenType.TEXT and self.peek().value == 'document':
                    self.advance()
                    self.expect(TokenType.RBRACE)
                    break
            # Handle macro definitions in preamble
            if self.peek().type in [TokenType.NEWCOMMAND, TokenType.RENEWCOMMAND]:
                macro_def = self.parse_macro_definition()
                if macro_def:
                    content.append(macro_def)
                continue
            self.advance()

        # Parse document content
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.END and self.peek(1).type == TokenType.LBRACE:
                save_pos = self.pos
                self.advance()
                self.advance()
                if self.peek().type == TokenType.TEXT and self.peek().value == 'document':
                    break
                self.pos = save_pos

            content.extend(self.parse_block())

        return Document(content)

    def parse_macro_definition(self) -> Optional[MacroDefNode]:
        """Parse \\newcommand or \\renewcommand"""
        token = self.advance()
        
        # Read macro name
        name = self.read_braced_text()
        if not name:
            return None
        
        # Remove leading backslash if present
        if name.startswith('\\'):
            name = name[1:]
        
        # Read number of parameters (optional)
        num_params = 0
        param_str = self.read_optional_arg()
        if param_str and param_str.isdigit():
            num_params = int(param_str)
        
        # Read definition - use the preserving function to keep backslashes
        definition = self.read_braced_text_preserving_commands()
        
        # Register macro
        self.macro_processor.define_macro(name, num_params, definition)
        
        return MacroDefNode(name, num_params, definition)

    def parse_block(self) -> List[ASTNode]:
        """Parse block-level elements"""
        self.skip_comments_and_newlines()
        token = self.peek()

        if token.type == TokenType.EOF:
            return []

        # Macro definitions
        if token.type in [TokenType.NEWCOMMAND, TokenType.RENEWCOMMAND]:
            macro_def = self.parse_macro_definition()
            return [macro_def] if macro_def else []

        # Sections
        if token.type in [TokenType.SECTION, TokenType.SUBSECTION, TokenType.SUBSUBSECTION]:
            return [self.parse_section()]

        # Environments
        if token.type == TokenType.BEGIN:
            return [self.parse_environment()]

        # Parse as inline content
        return self.parse_inline()

    def parse_section(self) -> Section:
        """Parse section headings"""
        token = self.advance()
        level_map = {
            TokenType.SECTION: 1,
            TokenType.SUBSECTION: 2,
            TokenType.SUBSUBSECTION: 3,
        }
        level = level_map[token.type]

        # Increment section counter
        counter_name = token.type.name.lower()
        self.symbol_table.increment_counter(counter_name)
        section_number = self.symbol_table.get_section_number()

        title_content = self.read_braced_content()
        title = self.content_to_text(title_content) if title_content else "Untitled"

        # Parse content until next section or EOF
        content = []
        while self.peek().type != TokenType.EOF:
            if self.peek().type in [TokenType.SECTION, TokenType.SUBSECTION, TokenType.SUBSUBSECTION]:
                break
            if self.peek().type == TokenType.END:
                break
            content.extend(self.parse_block())

        return Section(level, title, content, section_number)

    def parse_environment(self) -> ASTNode:
        """Parse LaTeX environments"""
        self.expect(TokenType.BEGIN)
        self.expect(TokenType.LBRACE)

        env_name_token = self.peek()
        env_name = env_name_token.value
        self.advance()
        self.expect(TokenType.RBRACE)

        # Handle different environment types
        if env_name == 'itemize' or env_name == 'enumerate':
            return self.parse_list(env_name)
        elif env_name == 'tabular':
            return self.parse_table()
        elif env_name in ['verbatim', 'lstlisting']:
            return self.parse_verbatim(env_name)
        elif env_name in ['quote', 'quotation', 'verse', 'center', 'flushleft', 'flushright']:
            return self.parse_block_environment(env_name)
        elif env_name == 'figure':
            self.symbol_table.increment_counter('figure')
            return self.parse_block_environment(env_name)
        elif env_name == 'table':
            self.symbol_table.increment_counter('table')
            return self.parse_block_environment(env_name)
        else:
            return self.parse_block_environment(env_name)

    def parse_list(self, list_type: str) -> ListNode:
        """Parse itemize or enumerate environments"""
        items = []
        current_item = None

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.END:
                if self.peek(1).type == TokenType.LBRACE:
                    save_pos = self.pos
                    self.advance()
                    self.advance()
                    if self.peek().type == TokenType.TEXT and self.peek().value == list_type:
                        self.advance()
                        if self.peek().type == TokenType.RBRACE:
                            self.advance()
                        
                        if current_item is not None:
                            items.append(current_item)
                        
                        return ListNode(list_type, items)
                    else:
                        self.pos = save_pos
                        self.advance()
                else:
                    self.advance()
            elif self.peek().type == TokenType.ITEM:
                if current_item is not None:
                    items.append(current_item)
                current_item = []
                self.advance()
                self.read_optional_arg()
            elif self.peek().type in [TokenType.COMMENT, TokenType.NEWLINE]:
                self.advance()
            elif current_item is not None:
                current_item.extend(self.parse_inline())
            else:
                self.advance()

        if current_item is not None:
            items.append(current_item)

        return ListNode(list_type, items)

    def parse_table(self) -> TableNode:
        """Parse tabular environment"""
        alignment = self.read_braced_content()
        align_str = self.content_to_text(alignment) if alignment else "l"

        rows = []
        current_row = []
        current_cell = []
        
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.END:
                if self.peek(1).type == TokenType.LBRACE:
                    save_pos = self.pos
                    self.advance()
                    self.advance()
                    if self.peek().type == TokenType.TEXT and self.peek().value == 'tabular':
                        self.advance()
                        if self.peek().type == TokenType.RBRACE:
                            self.advance()
                        
                        if current_cell:
                            current_row.append(current_cell)
                        if current_row:
                            rows.append(current_row)
                        
                        return TableNode(align_str, rows)
                    else:
                        self.pos = save_pos
                        self.advance()
                else:
                    self.advance()
            elif self.peek().type == TokenType.AMPERSAND:
                current_row.append(current_cell)
                current_cell = []
                self.advance()
            elif self.peek().type == TokenType.NEWLINE and self.peek().value == '\\\\':
                current_row.append(current_cell)
                if current_row:
                    rows.append(current_row)
                current_row = []
                current_cell = []
                self.advance()
            elif self.peek().type == TokenType.HLINE:
                self.advance()
            elif self.peek().type in [TokenType.COMMENT, TokenType.NEWLINE]:
                self.advance()
            else:
                current_cell.extend(self.parse_inline())

        if current_cell:
            current_row.append(current_cell)
        if current_row:
            rows.append(current_row)
        
        return TableNode(align_str, rows)

    def parse_verbatim(self, env_name: str) -> EnvironmentNode:
        """Parse verbatim or lstlisting environment"""
        content = []
        text = ""

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.END:
                if self.peek(1).type == TokenType.LBRACE:
                    save_pos = self.pos
                    self.advance()
                    self.advance()
                    if self.peek().type == TokenType.TEXT and self.peek().value == env_name:
                        self.advance()
                        if self.peek().type == TokenType.RBRACE:
                            self.advance()
                        
                        content.append(TextNode(text))
                        return EnvironmentNode(env_name, content)
                    else:
                        self.pos = save_pos
                        text += self.peek().value
                        self.advance()
                else:
                    text += self.peek().value
                    self.advance()
            else:
                text += self.peek().value
                self.advance()

        content.append(TextNode(text))
        return EnvironmentNode(env_name, content)

    def parse_block_environment(self, env_name: str) -> EnvironmentNode:
        """Parse block environments like quote, center, etc."""
        content = []

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.END:
                if self.peek(1).type == TokenType.LBRACE:
                    save_pos = self.pos
                    self.advance()
                    self.advance()
                    if self.peek().type == TokenType.TEXT and self.peek().value == env_name:
                        self.advance()
                        if self.peek().type == TokenType.RBRACE:
                            self.advance()
                        
                        return EnvironmentNode(env_name, content)
                    else:
                        self.pos = save_pos
                        content.extend(self.parse_block())
                else:
                    self.advance()
            else:
                content.extend(self.parse_block())

        return EnvironmentNode(env_name, content)

    def parse_inline(self) -> List[ASTNode]:
        """Parse inline elements"""
        token = self.peek()

        # Text - check for user-defined macros
        if token.type == TokenType.TEXT:
            text = token.value
            # Check if it's a macro call
            if text.startswith('\\'):
                macro_name = text[1:]
                if self.macro_processor.has_macro(macro_name):
                    self.advance()
                    # Read parameters
                    macro = self.macro_processor.macros[macro_name]
                    params = []
                    for _ in range(macro.num_params):
                        param_content = self.read_braced_content()
                        params.append(self.content_to_text(param_content))
                    
                    # Expand macro
                    expanded = self.macro_processor.expand_macro(macro_name, params)
                    
                    # Re-tokenize and parse the expansion
                    expanded_lexer = Lexer(expanded)
                    expanded_tokens = expanded_lexer.tokenize()
                    expanded_parser = Parser(expanded_tokens, self.macro_processor, self.symbol_table)
                    # Parse the expanded content
                    result = []
                    while expanded_parser.peek().type != TokenType.EOF:
                        result.extend(expanded_parser.parse_inline())
                    return result
            
            self.advance()
            return [TextNode(token.value)]

        # Newlines
        if token.type == TokenType.NEWLINE:
            self.advance()
            if token.value == '\n\n':
                return [TextNode('<br><br>')]
            return [TextNode(' ')]

        # Comments
        if token.type == TokenType.COMMENT:
            self.advance()
            return []

        # Math
        if token.type in [TokenType.MATH_INLINE, TokenType.MATH_DISPLAY]:
            return [self.parse_math()]

        # Labels
        if token.type == TokenType.LABEL:
            self.advance()
            label_content = self.read_braced_content()
            label_name = self.content_to_text(label_content)
            
            # Add to symbol table with current section/figure/table number
            current_number = self.symbol_table.get_section_number()
            if not current_number:
                current_number = str(self.symbol_table.get_counter('figure') or 
                                   self.symbol_table.get_counter('table') or "?")
            
            self.symbol_table.add_symbol(
                label_name, 'label', current_number, 
                token.line, token.col
            )
            
            return [LabelNode(label_name)]

        # References
        if token.type == TokenType.REF:
            self.advance()
            ref_content = self.read_braced_content()
            ref_name = self.content_to_text(ref_content)
            
            # Add reference to symbol table
            self.symbol_table.add_reference(ref_name, token.line, token.col)
            
            return [RefNode(ref_name)]

        # Formatting commands
        if token.type in [TokenType.TEXTBF, TokenType.TEXTIT, TokenType.UNDERLINE, TokenType.EMPH,
                         TokenType.TEXTTT, TokenType.TEXTRM, TokenType.TEXTSC]:
            format_map = {
                TokenType.TEXTBF: 'bold',
                TokenType.TEXTIT: 'italic',
                TokenType.UNDERLINE: 'underline',
                TokenType.EMPH: 'italic',
                TokenType.TEXTTT: 'monospace',
                TokenType.TEXTRM: 'roman',
                TokenType.TEXTSC: 'smallcaps',
            }
            self.advance()
            content = self.read_braced_content()
            if content:
                return [FormattedText(format_map[token.type], content)]
            else:
                return []

        # Font sizes
        if token.type in [TokenType.TINY, TokenType.SMALL, TokenType.LARGE, TokenType.HUGE]:
            size_map = {
                TokenType.TINY: 'tiny',
                TokenType.SMALL: 'small',
                TokenType.LARGE: 'large',
                TokenType.HUGE: 'huge',
            }
            self.advance()
            if self.peek().type == TokenType.LBRACE:
                content = self.read_braced_content()
                return [FormattedText(size_map[token.type], content)]
            else:
                return []

        # Links
        if token.type == TokenType.HREF:
            self.advance()
            url_content = self.read_braced_content()
            url = self.content_to_text(url_content) if url_content else "#"
            text_content = self.read_braced_content()
            if not text_content:
                text_content = [TextNode(url)]
            return [LinkNode(url, text_content)]

        if token.type == TokenType.URL:
            self.advance()
            url_content = self.read_braced_content()
            url = self.content_to_text(url_content) if url_content else "#"
            return [LinkNode(url, [TextNode(url)])]

        # Images
        if token.type == TokenType.INCLUDEGRAPHICS:
            self.advance()
            options_str = self.read_optional_arg()
            options = {}
            if options_str:
                options = {'options': options_str}
            path_content = self.read_braced_content()
            path = self.content_to_text(path_content)
            return [ImageNode(path, options)]

        # Commands that don't take arguments
        if token.type in [TokenType.CENTERING]:
            self.advance()
            return []

        # Color
        if token.type == TokenType.TEXTCOLOR:
            self.advance()
            color_content = self.read_braced_content()
            color = self.content_to_text(color_content) if color_content else "black"
            text_content = self.read_braced_content()
            if text_content:
                return [FormattedText(f'color:{color}', text_content)]
            else:
                return []

        # Just advance if we don't recognize it
        self.advance()
        return []

    def parse_math(self) -> MathNode:
        """Parse math mode"""
        delim = self.advance()
        is_display = delim.type == TokenType.MATH_DISPLAY

        math_content = ""
        if delim.value in ['$', '$$']:
            while self.peek().type != TokenType.EOF:
                if self.peek().type == TokenType.MATH_INLINE and delim.value == '$':
                    self.advance()
                    break
                elif self.peek().type == TokenType.MATH_DISPLAY and delim.value == '$$':
                    self.advance()
                    break
                # Preserve backslashes for math commands
                token = self.peek()
                if token.type == TokenType.MATH_COMMAND:
                    math_content += '\\' + token.value
                else:
                    math_content += token.value
                self.advance()
        elif delim.value in ['\\(', '\\[']:
            closing = '\\)' if delim.value == '\\(' else '\\]'
            while self.peek().type != TokenType.EOF:
                if self.peek().type == TokenType.MATH_INLINE and delim.value == '\\(' and self.peek().value == '\\)':
                    self.advance()
                    break
                elif self.peek().type == TokenType.MATH_DISPLAY and delim.value == '\\[' and self.peek().value == '\\]':
                    self.advance()
                    break
                # Preserve backslashes for math commands
                token = self.peek()
                if token.type == TokenType.MATH_COMMAND:
                    math_content += '\\' + token.value
                else:
                    math_content += token.value
                self.advance()

        # Number equations if display mode
        eq_number = 0
        if is_display:
            eq_number = self.symbol_table.increment_counter('equation')

        return MathNode(math_content, is_display, eq_number)

    def content_to_text(self, content: List[ASTNode]) -> str:
        """Convert AST content to plain text"""
        result = ""
        for node in content:
            if isinstance(node, TextNode):
                result += node.text
            elif isinstance(node, FormattedText):
                result += self.content_to_text(node.content)
        return result


# ============== HTML GENERATOR ==============

class HTMLGenerator:
    def __init__(self, symbol_table: SymbolTable):
        self.html = ""
        self.symbol_table = symbol_table

    def generate(self, ast: Document) -> str:
        """Generate HTML from AST"""
        html_parts = []
        html_parts.append(self.generate_header())

        for node in ast.content:
            html_parts.append(self.generate_node(node))

        html_parts.append(self.generate_footer())
        return ''.join(html_parts)

    def generate_header(self) -> str:
        """Generate HTML header with styling and MathJax"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX Document</title>
    
    <!-- MathJax Configuration -->
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true,
            tags: 'ams'
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 60px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Arial', sans-serif;
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
        }
        
        h1 { font-size: 2.5em; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
        h2 { font-size: 2em; border-bottom: 1px solid #95a5a6; padding-bottom: 0.2em; }
        h3 { font-size: 1.5em; }
        h4 { font-size: 1.2em; }
        
        .section-number {
            color: #3498db;
            font-weight: bold;
            margin-right: 0.5em;
        }
        
        p {
            margin-bottom: 1em;
            text-align: justify;
        }
        
        strong, b {
            font-weight: bold;
            color: #2c3e50;
        }
        
        em, i {
            font-style: italic;
        }
        
        u {
            text-decoration: underline;
        }
        
        code, .monospace {
            font-family: 'Courier New', monospace;
            background: #f8f8f8;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        
        .smallcaps {
            font-variant: small-caps;
        }
        
        .tiny { font-size: 0.7em; }
        .small { font-size: 0.85em; }
        .large { font-size: 1.2em; }
        .huge { font-size: 1.5em; }
        
        ul, ol {
            margin-left: 2em;
            margin-bottom: 1em;
        }
        
        li {
            margin-bottom: 0.5em;
        }
        
        table {
            border-collapse: collapse;
            margin: 1em 0;
            width: 100%;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        tr:hover {
            background-color: #e8f4f8;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .ref-link {
            color: #e74c3c;
            font-weight: 500;
            cursor: pointer;
        }
        
        .label-target {
            position: relative;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }
        
        .quote, .quotation {
            margin: 1.5em 2em;
            padding: 1em;
            background: #f9f9f9;
            border-left: 4px solid #3498db;
            font-style: italic;
        }
        
        .verse {
            margin: 1.5em 2em;
            font-style: italic;
            white-space: pre-line;
        }
        
        .center {
            text-align: center;
        }
        
        .flushleft {
            text-align: left;
        }
        
        .flushright {
            text-align: right;
        }
        
        pre, .verbatim {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
            margin: 1em 0;
            font-family: 'Courier New', monospace;
        }
        
        .math-display {
            margin: 1em 0;
            overflow-x: auto;
            text-align: center;
        }
        
        .equation-number {
            float: right;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
                padding: 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
'''

    def generate_footer(self) -> str:
        """Generate HTML footer"""
        return '''
    </div>
</body>
</html>'''

    def generate_node(self, node: ASTNode) -> str:
        """Generate HTML for a single AST node"""
        if isinstance(node, TextNode):
            return self.escape_html(node.text)

        elif isinstance(node, Section):
            content_html = ''.join(self.generate_node(n) for n in node.content)
            number_html = f'<span class="section-number">{node.number}</span>' if node.number else ''
            return f'<h{node.level} id="section-{node.number}">{number_html}{self.escape_html(node.title)}</h{node.level}>\n{content_html}\n'

        elif isinstance(node, FormattedText):
            content_html = ''.join(self.generate_node(n) for n in node.content)
            if node.format_type == 'bold':
                return f'<strong>{content_html}</strong>'
            elif node.format_type == 'italic':
                return f'<em>{content_html}</em>'
            elif node.format_type == 'underline':
                return f'<u>{content_html}</u>'
            elif node.format_type == 'monospace':
                return f'<code>{content_html}</code>'
            elif node.format_type == 'smallcaps':
                return f'<span class="smallcaps">{content_html}</span>'
            elif node.format_type == 'tiny':
                return f'<span class="tiny">{content_html}</span>'
            elif node.format_type == 'small':
                return f'<span class="small">{content_html}</span>'
            elif node.format_type == 'large':
                return f'<span class="large">{content_html}</span>'
            elif node.format_type == 'huge':
                return f'<span class="huge">{content_html}</span>'
            elif node.format_type.startswith('color:'):
                color = node.format_type.split(':')[1]
                return f'<span style="color: {color}">{content_html}</span>'
            else:
                return content_html

        elif isinstance(node, MathNode):
            if node.is_display:
                eq_num = f'<span class="equation-number">({node.number})</span>' if node.number else ''
                return f'<div class="math-display">{eq_num}$$\n{node.content}\n$$</div>\n'
            else:
                return f'${node.content}$'

        elif isinstance(node, ListNode):
            tag = 'ul' if node.list_type == 'itemize' else 'ol'
            items_html = []
            for item in node.items:
                item_content = ''.join(self.generate_node(n) for n in item)
                items_html.append(f'<li>{item_content}</li>')
            return f'<{tag}>\n' + '\n'.join(items_html) + f'\n</{tag}>\n'

        elif isinstance(node, TableNode):
            rows_html = []
            for i, row in enumerate(node.rows):
                cells_html = []
                for cell in row:
                    cell_content = ''.join(self.generate_node(n) for n in cell)
                    tag = 'th' if i == 0 else 'td'
                    cells_html.append(f'<{tag}>{cell_content}</{tag}>')
                rows_html.append('<tr>' + ''.join(cells_html) + '</tr>')
            return '<table>\n' + '\n'.join(rows_html) + '\n</table>\n'

        elif isinstance(node, LinkNode):
            text_html = ''.join(self.generate_node(n) for n in node.text)
            return f'<a href="{self.escape_html(node.url)}">{text_html}</a>'

        elif isinstance(node, ImageNode):
            alt = node.options.get('alt', 'Image')
            return f'<img src="{self.escape_html(node.path)}" alt="{alt}">\n'

        elif isinstance(node, EnvironmentNode):
            content_html = ''.join(self.generate_node(n) for n in node.content)
            if node.env_type in ['quote', 'quotation', 'verse', 'center', 'flushleft', 'flushright']:
                return f'<div class="{node.env_type}">\n{content_html}\n</div>\n'
            elif node.env_type in ['verbatim', 'lstlisting']:
                return f'<pre class="verbatim">{content_html}</pre>\n'
            else:
                return f'<div class="environment-{node.env_type}">\n{content_html}\n</div>\n'

        elif isinstance(node, LabelNode):
            # Labels are invisible anchors
            return f'<span id="label-{node.name}" class="label-target"></span>'

        elif isinstance(node, RefNode):
            # Look up the reference in symbol table
            symbol = self.symbol_table.lookup(node.name)
            if symbol:
                ref_text = str(symbol.value)
                return f'<a href="#label-{node.name}" class="ref-link">{ref_text}</a>'
            else:
                return f'<span class="ref-link" style="color: #e74c3c;">??</span>'

        elif isinstance(node, MacroDefNode):
            # Macro definitions are invisible
            return ''

        return ''

    def escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


# ============== MAIN COMPILER ==============

def compile_latex_to_html(latex_code: str) -> Tuple[str, SymbolTable]:
    """Main function to compile LaTeX to HTML"""
    try:
        # Initialize symbol table and macro processor
        symbol_table = SymbolTable()
        macro_processor = MacroProcessor()

        # Lexical analysis
        lexer = Lexer(latex_code)
        tokens = lexer.tokenize()

        # Parsing
        parser = Parser(tokens, macro_processor, symbol_table)
        ast = parser.parse()

        # HTML generation
        generator = HTMLGenerator(symbol_table)
        html = generator.generate(ast)

        return html, symbol_table
    except SyntaxError as e:
        return f"<pre style='color: red; background: #fff0f0; padding: 20px; border: 2px solid red;'>Syntax Error: {e}</pre>", SymbolTable()
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"<pre style='color: red; background: #fff0f0; padding: 20px; border: 2px solid red;'>Unexpected error: {e}\n\nTraceback:\n{error_trace}</pre>", SymbolTable()


def create_download_link(html_content: str, filename: str):
    """Create a download link for HTML content"""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="display: inline-block; padding: 12px 28px; background: linear-gradient(135deg, #e94560 0%, #c23950 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4); transition: all 0.3s ease;">Download HTML</a>'
    return href


def main():
    st.set_page_config(
        page_title="LaTeX to HTML Converter",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background: #1a1a2e;
        }

        .app-header {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid #2a2a4e;
        }

        .app-header h1 {
            color: #e94560;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        .app-header p {
            color: #a8b2d1;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .feature-card {
            background: #16213e;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
            border-left: 4px solid #e94560;
        }

        .stButton > button {
            background: linear-gradient(135deg, #e94560 0%, #c23950 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(233, 69, 96, 0.6);
            background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
        }

        h1, h2, h3, h4, h5, h6 {
            color: #e94560 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="app-header">
            <h1>LaTeX to HTML Converter</h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("Features")

        with st.expander("Document Structures"):
            st.markdown("""
            - Sections: `\\section{Title}`
            - Lists: `itemize`, `enumerate`
            - Tables: `tabular` environment
            - Figures & images
            - Quotes & code blocks
            """)

        with st.expander("Special Features"):
            st.markdown("""
            - Colors: `\\textcolor{red}{text}`
            - Links: `\\href{url}{text}`
            - Font sizes: `\\large`, `\\huge`
            - Alignment: `center`, `flushleft`
            """)
        
        with st.expander("Cross-References"):
            st.markdown("""
            - **Labels**: `\\label{sec:intro}`
            - **References**: `\\ref{sec:intro}`
            - Automatic numbering for sections
            - Clickable hyperlinks
            """)

        with st.expander("Macro Expansion"):
            st.markdown("""
            - **Define**: `\\newcommand{\\mycmd}{text}`
            - **With params**: `\\newcommand{\\mycmd}[2]{#1 and #2}`
            - Recursive expansion
            - Built-in macros: `\\LaTeX`, `\\TeX`
            """)

        with st.expander("Text Formatting"):
            st.markdown("""
            - **Bold**: `\\textbf{text}`
            - *Italic*: `\\textit{text}`
            - <u>Underline</u>: `\\underline{text}`
            - `Code`: `\\texttt{text}`
            """, unsafe_allow_html=True)

        with st.expander("Math Expressions"):
            st.markdown("""
            - Inline: `$x^2 + y^2$`
            - Display: `$$E = mc^2$$`
            - Auto-numbered equations
            """)

        st.markdown("---")

    # Main content
    latex_content = None

    st.markdown("Upload Your LaTeX File")

    uploaded_file = st.file_uploader(
        "Choose a .tex file",
        type=['tex'],
        help="Upload a LaTeX (.tex) file to convert to HTML"
    )

    if uploaded_file is not None:
        latex_content = uploaded_file.read().decode('utf-8')
        st.success(f"File **{uploaded_file.name}** uploaded successfully!")

        with st.expander("Preview LaTeX Source", expanded=False):
            st.code(latex_content[:1000] + ("..." if len(latex_content) > 1000 else ""), language='latex')

    # Convert button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        convert_button = st.button("Convert to HTML", type="primary", use_container_width=True)

    if convert_button:
        if not latex_content:
            st.error("Please upload a LaTeX file first!")
        else:
            with st.spinner("Converting your document..."):
                html_output, symbol_table = compile_latex_to_html(latex_content)

                if "Error:" in html_output and html_output.startswith("<pre"):
                    st.error("Compilation Error")
                    st.markdown(html_output, unsafe_allow_html=True)
                else:
                    st.success("Conversion successful!")

                    # Show symbol table info
                    if symbol_table.symbols or symbol_table.warnings:
                        with st.expander("Symbol Table Information", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Defined Symbols:**")
                                if symbol_table.symbols:
                                    for name, symbol in symbol_table.symbols.items():
                                        st.text(f"• {name} ({symbol.symbol_type}): {symbol.value}")
                                        if symbol.references:
                                            st.text(f"  Referenced {len(symbol.references)} time(s)")
                                else:
                                    st.text("No symbols defined")
                            
                            with col2:
                                st.markdown("**Counters:**")
                                for counter, value in symbol_table.counters.items():
                                    if value > 0:
                                        st.text(f"• {counter}: {value}")
                                
                                if symbol_table.warnings:
                                    st.markdown("**Warnings:**")
                                    for warning in symbol_table.warnings:
                                        st.warning(warning)

                    st.markdown("---")

                    # Create tabs
                    tab1, tab2 = st.tabs(["HTML Preview", "HTML Code"])

                    with tab1:
                        st.markdown("### Preview")
                        st.components.v1.html(html_output, height=800, scrolling=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            st.markdown(create_download_link(html_output, "output.html"), unsafe_allow_html=True)

                    with tab2:
                        st.markdown("### HTML Source Code")
                        st.code(html_output, language='html', line_numbers=True)
                        


if __name__ == "__main__":
    main()
