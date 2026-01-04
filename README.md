# LaTeX to HTML Compiler

A custom LaTeX compiler that converts `.tex` files to HTML with full support for cross-references, macros, and mathematical expressions.

## Features

### Document Structure
- **Sections**: `\section{}`, `\subsection{}`, `\subsubsection{}`
- **Lists**: `itemize`, `enumerate`, `description` environments
- **Tables**: `tabular` environment with alignment (`l`, `c`, `r`)
- **Figures**: `\includegraphics` and `figure` environment
- **Quotes**: `quote`, `quotation`, `verse` environments

### Text Formatting
- **Bold**: `\textbf{text}`
- **Italic**: `\textit{text}` or `\emph{text}`
- **Underline**: `\underline{text}`
- **Monospace**: `\texttt{text}`
- **Small caps**: `\textsc{text}`
- **Font sizes**: `\tiny`, `\small`, `\large`, `\huge`

### Colors
```latex
\textcolor{red}{colored text}
\textcolor{#FF5733}{hex colors supported}
```

### Mathematics
```latex
Inline math: $E = mc^2$
Display math: $$\int_0^\infty e^{-x^2} dx$$
```

### Cross-References
```latex
\section{Introduction}
\label{sec:intro}

See Section \ref{sec:intro} for details.
```
References automatically resolve to correct section numbers and become clickable links in HTML.

### Custom Macros
```latex
% No parameters
\newcommand{\myname}{John Doe}

% With parameters
\newcommand{\important}[1]{\textbf{\textcolor{red}{#1}}}

% Usage
\myname
\important{critical info}
```

### Hyperlinks
```latex
\href{https://example.com}{link text}
\url{https://example.com}
```

### Code Blocks
```latex
\begin{verbatim}
def hello():
    print("Hello, world!")
\end{verbatim}
```

## Usage

### Running the Compiler

```bash
streamlit run latex_to_html_compiler.py
```

### Web Interface
1. Upload your `.tex` file
2. Click "Convert to HTML"
3. Preview the rendered HTML
4. Download the output

### Symbol Table
The compiler provides a symbol table showing:
- Defined labels and their locations
- Section/equation counters
- Reference usage statistics
- Compilation warnings

## Example

```latex
\documentclass{article}

\newcommand{\highlight}[1]{\textit{\textcolor{blue}{#1}}}

\begin{document}

\section{Introduction}
\label{sec:intro}

This is \highlight{highlighted text} with a reference to \ref{sec:math}.

\section{Math}
\label{sec:math}

Einstein's famous equation: $$E = mc^2$$

\end{document}
```

## Supported Environments

| Environment | Description |
|-------------|-------------|
| `itemize` | Bulleted lists |
| `enumerate` | Numbered lists |
| `tabular` | Tables |
| `figure` | Figures with captions |
| `center` | Centered content |
| `verbatim` | Code blocks |

## Limitations

- Single-file compilation only (no `\input` or `\include`)
- Limited subset of LaTeX packages
- Math rendering uses MathJax (requires internet connection)
- Some advanced LaTeX features not supported


