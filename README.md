# Natural Language Processing with Deep Learning 2023/2024

This repository contains slide decks for the course "L.079.05551 Natural Language Processing with Deep Learning" (Paderborn University, winter term 2023/2024).

This course is lectured by [Prof. Dr. Ivan Habernal](https://www.trusthlt.org).

The slides are available as PDF as well as LaTeX source code (we've used Beamer because typesetting mathematics in PowerPoint or similar tools is painful). See the instructions below if you want to compile the slides yourselves.

![Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/CC_BY-SA_icon.svg/88px-CC_BY-SA_icon.svg.png)

The content is licensed under [Creative Commons CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) which means that you can re-use, adapt, modify, or publish it further, provided you keep the license and give proper credits.

**Note:** The following content is continuously updated as the winter term progresses.

## YouTube Playlist

Subscribe the YouTube playlist to get updates on new lectures: https://youtube.com/playlist?list=PL6WLGVNe6ZcA4gUr5MaAKdrGxYzYAETK3

## 2023-10-13 Lecture 01: NLP tasks and evaluation

* [Slides as PDF](/lecture01/pdf/nlpwdl2023-lecture01-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=QQwJdUeONCI)
* Exercise: See the PDF (including LaTeX source) and Python code under `exercises/ex01`
 
## FAQ

* What are some essential pre-requisites?
  * Math: Derivatives and partial derivatives. We cover them in Lecture 2. If you need more, I would recommend these sources:
    * *Jeremy Kun: A Programmer's Introduction to Mathematics.* Absolutely amazing book. Pay-what-you-want for the PDF book. https://pimbook.org/
    * *Deisenroth, A. Aldo Faisal, and Cheng Soon Ong: Mathematics for Machine Learning*. Excellent resource, freely available. Might be a bit dense. https://mml-book.github.io/
* Where do I find the code for plotting the functions?
  * Most of the plots are generated in Python/Jupyter (in Colab). The links are included as comments in the respective LaTeX sources for the slides.

## Compiling slides to PDF

If you run a linux distribution (e.g., Ubuntu 20.04 and newer), all packages are provided as part of `texlive`. Install the following packages

```plain
$ sudo apt-get install texlive-latex-recommended texlive-pictures texlive-latex-extra \
texlive-fonts-extra texlive-bibtex-extra texlive-humanities texlive-science \
texlive-luatex biber wget -y
```

Install Fira Sans fonts required by the beamer template locally

```plain
$ wget https://github.com/mozilla/Fira/archive/refs/tags/4.106.zip -O 4.106.zip \
&& unzip -o 4.106.zip && mkdir -p ~/.fonts/FiraSans && cp Fira-4.106/otf/Fira* \
~/.fonts/FiraSans/ && rm -rf Fira-4.106 && rm 4.106.zip && fc-cache -f -v && mktexlsr
```

Compile each lecture's slides using ``lualatex``

```plain
$ lualatex dl4nlp2023-lecture*.tex && biber dl4nlp2023-lecture*.bcf && \
lualatex dl4nlp2023-lecture*.tex && lualatex dl4nlp2023-lecture*.tex
```

### Compiling slides using Docker

If you don't run a linux system or don't want to mess up your latex packages, I've tested compiling the slides in a Docker.

Install Docker ( https://docs.docker.com/engine/install/ )

Create a folder to which you clone this repository (for example, `$ mkdir -p /tmp/slides`)

Run Docker with Ubuntu 20.04 interactively; mount your slides directory under `/mnt` in this Docker container

```plain
$ docker run -it --rm --mount type=bind,source=/tmp/slides,target=/mnt \
ubuntu:20.04 /bin/bash
```

Once the container is running, update, install packages and fonts as above

```plain
# apt-get update && apt-get dist-upgrade -y && apt-get install texlive-latex-recommended \
texlive-pictures texlive-latex-extra texlive-fonts-extra texlive-bibtex-extra \
texlive-humanities texlive-science texlive-luatex biber wget -y
```

Fonts

```plain
# wget https://github.com/mozilla/Fira/archive/refs/tags/4.106.zip -O 4.106.zip \
&& unzip -o 4.106.zip && mkdir -p ~/.fonts/FiraSans && cp Fira-4.106/otf/Fira* \
~/.fonts/FiraSans/ && rm -rf Fira-4.106 && rm 4.106.zip && fc-cache -f -v && mktexlsr
```

And compile

```plain
# cd /mnt/dl4nlp/latex/lecture01
# lualatex dl4nlp2023-lecture*.tex && biber dl4nlp2023-lecture*.bcf && \
lualatex dl4nlp2023-lecture*.tex && lualatex dl4nlp2023-lecture*.tex
```

which generates the PDF in your local folder (e.g, `/tmp/slides`).

### Compiling handouts

Run the script `compile-pdf.sh` to produce both handouts as well as unfolding PDFs used in the lecture.
