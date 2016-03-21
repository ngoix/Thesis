%%
%% This is file `ecsthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% ecsdocs.dtx  (with options: `thesis')
%% 
%% Copyright (C) 2001 by Steve R. Gunn
%% 
%% This file is part of the ECSDocs class distribution
%% 
\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{ecsthesis}
              [2003/25/04 v1.6
   LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{11pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand\bhrule{\typeout{------------------------------------------------------------------------------}}
\newcommand\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\year}
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}
\usepackage{vmargin}
\setmarginsrb           { 1.5in}  % left margin
                        { 0.6in}  % top margin
                        { 1.0in}  % right margin
                        { 0.8in}  % bottom margin
                        {  20pt}  % head height
                        {0.25in}  % head sep
                        {   9pt}  % foot height
                        { 0.3in}  % foot sep
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\btypeout{\thechapter.\quad #1}\markboth{\@chapapp\ \thechapter.\quad #1}{\@chapapp\ \thechapter.\quad #1}}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace}
\theoremstyle{plain}
\newtheorem{example}{Example}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{axiom}[theorem]{Axiom}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\usepackage[small,sc]{caption}
\setlength{\captionmargin}{20pt}
\newcommand{\fref}[1]{Figure~\ref{#1}}
\newcommand{\tref}[1]{Table~\ref{#1}}
\newcommand{\eref}[1]{Equation~\ref{#1}}
\newcommand{\cref}[1]{Chapter~\ref{#1}}
\newcommand{\sref}[1]{Section~\ref{#1}}
\newcommand{\aref}[1]{Appendix~\ref{#1}}
\newcommand{\defref}[1]{Definition~\ref{#1}}
\newcommand{\thref}[1]{Theorem~\ref{#1}}
\newcommand{\lref}[1]{Lemma~\ref{#1}}
\newcommand{\pref}[1]{Part~\ref{#1}}
\newcommand{\exref}[1]{Example~\ref{#1}}
\newcommand{\propref}[1]{Proposition~\ref{#1}}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}
\usepackage{chngcntr}  % to use \counterwithout{section}{chapter}
\usepackage{graphicx}
\usepackage{epstopdf}
\usepackage[scriptsize]{subfigure}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{listings}
%\usepackage{lstpatch}
\usepackage{xcolor}

\lstdefinestyle{matlab} {
        language=Matlab,
        keywordstyle=\color{blue},
        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
        stringstyle=\color[rgb]{0.7,0,0} }
\definecolor{linkcol}{rgb}{0,0,0.4}
\definecolor{urlcol}{rgb}{0.4,0,0}
\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
   bookmarksopenlevel=0,bookmarksnumbered=true,hypertexnames=false,
   colorlinks,linkcolor={linkcol},citecolor={linkcol},urlcolor={urlcol},
   pdfstartview={FitV},unicode,breaklinks=true,hypertexnames=true]{hyperref}
\pdfstringdefDisableCommands{
   \let\\\space
}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}
\newcommand*{\titre}[1]{\def\titrename{#1}}
\authors     {}
\subject     {}
\keywords    {}
\renewcommand\maketitle{
  \btypeout{Title Page}
  \hypersetup{pdftitle={\@title}}
  \hypersetup{pdfsubject=\subjectname}
  \hypersetup{pdfauthor=\authornames}
  \hypersetup{pdfkeywords=\keywordnames}
  \thispagestyle{empty}
  \begin{titlepage}
  \begin{center}
\vfil\null
{\'E}cole Doctorale ED130 ``Informatique, t\'el\'ecommunications et \'electronique de Paris''
 \begin{tabular}{ccc}
   ~\\
   ~\\
   \includegraphics[height=1.7cm]{images/Front/logo_TELECOM.png}\quad&\quad
   \includegraphics[height=1.7cm]{images/Front/logo_ENSCACHAN.png}\quad&\quad
   \includegraphics[height=1.7cm]{images/Front/logo_CNRS.jpg}\\
   ~\\
  \end{tabular}
    \par\vspace*{\baselineskip}
    \doublespacing
    {\LARGE{\bf \@title}}\\
    \vspace*{0.2cm}
    {\LARGE{\bf\sffamily ---}}\\
    \vspace*{0.2cm}
    {\LARGE{\bf \titrename}}\\
    \vspace*{0.5cm}
\singlespacing
    Th{\`e}se 
    pour obtenir le grade de docteur d\'elivr\'e par\\[\baselineskip]
    {\Large\bfseries TELECOM PARISTECH}\\[\baselineskip]
    \textbf{Sp\'ecialit\'e~ ``Signal et Images''}\\[\baselineskip]
    \emph{pr{\'e}sent{\'e}e et soutenue publiquement par}\\[\baselineskip]
    {\Large\bfseries Nicolas\ \textsc{Goix}}\\[0.5\baselineskip]
    le 1er Octobre 2016
    \singlespacing
        LTCI, CNRS, T\'el\'ecom ParisTech, Universit\'e Paris-Saclay, 75013, Paris, France\\
        \par\vspace*{2.5\baselineskip}
\hrule
\par\vspace*{0.3cm}
    \textbf{Jury~:} \\[0.5\baselineskip]
      \begin{tabular}{lll}
        St\'ephane \textbf{Boucheron} & Professeur, Universit\'e Paris Diderot & Rapporteur\\
        St\'ephan \textbf{Cl\'emen\c{c}on} & Professeur, T\'el\'ecom ParisTech & Directeur\\
        Luc \textbf{Devroye} & Professeur, McGill University  & Examinateur\\
        Anne \textbf{Sabourin} & Professeur, T\'el\'ecom ParisTech & Co-directeur\\
        Johan \textbf{Segers} & Professeur, Universit\'e de Lille & Examinateur\\
        Jean-Philippe \textbf{Vert} & Professeur, Mines ParisTech & Rapporteur\\
      \end{tabular}
\par\vspace*{0.5cm}
\hrule
  \end{center}
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \cleardoublepage
}

\usepackage{ifthen}
 
\newcommand{\newevenside}{
        \ifthenelse{\isodd{\thepage}}{
        \newpage
        \phantom{placeholder} % doesn't appear on page
        \thispagestyle{empty} % if want no header/footer
        \newpage
        }{\newpage}
}

\newenvironment{abstract}
{
  \newevenside
  \btypeout{Abstract}
  \thispagestyle{empty}

}
{}
\addtocounter{secnumdepth}{1}
\setcounter{tocdepth}{1}
\newcounter{dummy}
\newcommand\addtotoc[1]{
\refstepcounter{dummy}
\addcontentsline{toc}{chapter}{#1}}
\renewcommand\tableofcontents{
\btypeout{Table of Contents}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
   \cleardoublepage
}\end{spacing}
}
\renewcommand\listoffigures{
\btypeout{List of Figures}
\addtotoc{List of Figures}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\listoftables{
\btypeout{List of Tables}
\addtotoc{List of Tables}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\newcommand\listsymbolname{Nomenclature}
\usepackage{longtable}
\newcommand\listofsymbols[2]{
\btypeout{\listsymbolname}
\addtotoc{\listsymbolname}
    \chapter*{\listsymbolname
      \@mkboth{
          \MakeUppercase\listsymbolname}{\MakeUppercase\listsymbolname}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}
\newenvironment{acknowledgements}
{
  \btypeout{Acknowledgements}
  \thispagestyle{plain}
        \chapter*{Remerciements
        \@mkboth{
           \MakeUppercase{Remerciements}}{\MakeUppercase{Remerciements}}}

}
{
  \vfil\vfil\vfil\null
  \cleardoublepage
}
\newenvironment{listpublis}
{
  \cleardoublepage
  \btypeout{List of Publications}
  \addtotoc{List of Publications}
  \thispagestyle{plain}
        \chapter*{List of Publications
        \@mkboth{
           \MakeUppercase{List of Publications}}{\MakeUppercase{List of Publications}}}

}
{
  \vfil\vfil\vfil\null
  \cleardoublepage
}
\newcommand\dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}
\renewcommand\backmatter{
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \addtotoc{\bibname}
  \btypeout{\bibname}
  \@mainmatterfalse}
\endinput
%%
%% End of file `ecsthesis.cls'.

