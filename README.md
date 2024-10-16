# Natural Language Processing with Deep Learning 2024/2025

This repository contains **slide decks**, **programming exercises**, and links to **recorded lectures videos** for the course "Natural Language Processing with Deep Learning" (Ruhr University Bochum, winter term 2024/2025).

This course is lectured by [Prof. Dr. Ivan Habernal](https://www.trusthlt.org).

The slides are available as PDF as well as LaTeX source code (we've used Beamer because typesetting mathematics in PowerPoint or similar tools is painful). See the instructions below if you want to compile the slides yourselves.

![Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/CC_BY-SA_icon.svg/88px-CC_BY-SA_icon.svg.png)

The content is licensed under [Creative Commons CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) which means that you can re-use, adapt, modify, or publish it further, provided you keep the license and give proper credits.

**Note:** The following content is continuously updated as the winter term progresses.

To see the previous full **2023/24** course content, checkout the [latest 2024 commit](https://github.com/trusthlt/nlp-with-deep-learning-lectures/tree/ae2364ac136e9852a6992f17a90771f0a5474bfb).

## YouTube Playlist

Subscribe the YouTube playlist to get updates on new lectures: (TO BE DECIDED/UPDATED)

## Lectures and exercises

### Lecture 01: NLP tasks and evaluation

2024-10-17

* [Slides as PDF](/lecture01/pdf/nlpwdl2024-lecture01-handout.pdf)
* [YouTube recording TBD](https://www.youtube.com/watch?v=xxxx)

### Exercise 01: Classification evaluation, text generation evaluation (TO BE UPDATED)

2024-MM-DD

* See the PDF (including LaTeX source) and Python code under [exercises/ex01](exercises/ex01)



## FAQ

* What are some essential pre-requisites?
  * Math: Derivatives and partial derivatives. We cover them in Lecture 2. If you need more, I would recommend these sources:
    * *Jeremy Kun: A Programmer's Introduction to Mathematics.* Absolutely amazing book. Pay-what-you-want for the PDF book. https://pimbook.org/
    * *Deisenroth, A. Aldo Faisal, and Cheng Soon Ong: Mathematics for Machine Learning*. Excellent resource, freely available. Might be a bit dense. https://mml-book.github.io/
* Where do I find the code for plotting the functions?
  * Most of the plots are generated in Python/Jupyter (in Colab). The links are included as comments in the respective LaTeX sources for the slides.

## Compiling slides to PDF

If you run a linux distribution (e.g., Ubuntu 22.04 and newer), all packages are provided as part of `texlive`. Install the following packages

```plain
$ sudo apt-get install texlive-latex-recommended texlive-pictures texlive-latex-extra \
texlive-fonts-extra texlive-bibtex-extra texlive-humanities texlive-science \
texlive-luatex biber wget -y
```

* Install RUB fonts
  * I've prepared a shell script which downloads the TTF files and installs them

```bash
$ chmod +x install-rub-fonts.sh
$ ./install-rub-fonts.sh
```

Run the script `compile-pdf.sh` in each lecture's folder to produce both handouts as well as unfolding PDFs used in the lecture.

### Compiling slides using Docker

If you don't run a linux system or don't want to mess up your latex packages, I've tested compiling the slides in a Docker.

Install Docker ( https://docs.docker.com/engine/install/ )

Create a folder to which you clone this repository (for example, `$ mkdir -p /tmp/slides`)

Run Docker with Ubuntu 22.04 interactively; mount your slides directory under `/mnt` in this Docker container

```plain
$ docker run -it --rm --mount type=bind,source=/tmp/slides,target=/mnt \
ubuntu:22.04 /bin/bash
```

Once the container is running, update, install packages and fonts as above

```plain
# apt-get update && apt-get dist-upgrade -y && apt-get install texlive-latex-recommended \
texlive-pictures texlive-latex-extra texlive-fonts-extra texlive-bibtex-extra \
texlive-humanities texlive-science texlive-luatex biber wget -y
```

Install RUB fonts as above

Run the script `compile-pdf.sh` in each lecture's folder to produce both handouts as well as unfolding PDFs used in the lecture, which generates the PDFs in your local folder (e.g, `/tmp/slides`).

