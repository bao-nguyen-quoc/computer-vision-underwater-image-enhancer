# Surveys

[A Comprehensive Survey on Underwater Image Enhancement Based on Deep Learning](https://arxiv.org/pdf/2405.19684)
[Visual enhancement and 3D representation for underwater scenes](https://arxiv.org/pdf/2505.01869)
[Springer Visual enhancement and 3D representation for underwater scenes](https://link.springer.com/article/10.1007/s10462-026-11597-4)

# Datasets & Benchmark

[An Underwater Image Enhancement Benchmark Dataset and Beyond](https://arxiv.org/pdf/1901.05495)

# CNN

[Shallow-UWnet: Compressed models for underwater image enhancement](https://arxiv.org/pdf/2101.02073)

# GAN

[Fast Underwater Image Enhancement for Improved Visual Perception](https://arxiv.org/pdf/1903.09766)
[ieee Fast Underwater Image Enhancement for Improved Visual Perception](https://ieeexplore.ieee.org/document/9001231)

# Prepare template

```LaTeX
\documentclass[conference,a4paper]{IEEEtran}

% ---- Encoding & language ----
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
% Nếu viết tiếng Việt thì đổi T1 thành T5 và thêm: \usepackage[vietnamese]{babel}

% ---- Math, graphics, tables ----
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{algorithmic}
\usepackage{algorithm}

% ---- Citation & links (hyperref nên đặt gần cuối) ----
\usepackage{cite}
\usepackage[hidelinks]{hyperref}

\graphicspath{{figures/}}
\hyphenation{op-tical net-works semi-conduc-tor}

\begin{document}

\title{A Comparative Study of Deep Learning Pipelines\\for Underwater Image Enhancement}

\author{
  \IEEEauthorblockN{Họ Tên 1, Họ Tên 2, Họ Tên 3}
  \IEEEauthorblockA{Faculty of Computer Science, University Name\\
  City, Country\\
  email@domain.edu}
}

\maketitle

\begin{abstract}
...
\end{abstract}

\begin{IEEEkeywords}
underwater image enhancement, deep learning, GAN, transformer, benchmark
\end{IEEEkeywords}

\section{Introduction}
\section{Related Work}
\section{Methods}
\subsection{Shallow-UWnet}
\subsection{FUnIE-GAN}
\subsection{U-shape Transformer}
\section{Experimental Setup}
\subsection{Datasets}
\subsection{Training Protocol}
\subsection{Evaluation Metrics}
\section{Results and Discussion}
\section{Conclusion}

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
```