# Natural Language Processing with Deep Learning 2023/2024

This repository contains **slide decks**, **programming exercises**, and links to **recorded lectures videos** for the course "L.079.05551 Natural Language Processing with Deep Learning" (Paderborn University, winter term 2023/2024).

This course is lectured by [Prof. Dr. Ivan Habernal](https://www.trusthlt.org).

The slides are available as PDF as well as LaTeX source code (we've used Beamer because typesetting mathematics in PowerPoint or similar tools is painful). See the instructions below if you want to compile the slides yourselves.

![Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/CC_BY-SA_icon.svg/88px-CC_BY-SA_icon.svg.png)

The content is licensed under [Creative Commons CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) which means that you can re-use, adapt, modify, or publish it further, provided you keep the license and give proper credits.

**Note:** The following content is continuously updated as the winter term progresses.

## YouTube Playlist

Subscribe the YouTube playlist to get updates on new lectures: https://www.youtube.com/playlist?list=PL6WLGVNe6ZcB00apoxMtj7WSUOlpm2Xvl

## Lectures and exercises

### Lecture 01: NLP tasks and evaluation

2023-10-13

* [Slides as PDF](/lecture01/pdf/nlpwdl2023-lecture01-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=QQwJdUeONCI)

### Exercise 01: Classification evaluation, text generation evaluation

2023-10-19

* See the PDF (including LaTeX source) and Python code under [exercises/ex01](exercises/ex01)

### Lecture 02: Mathematical foundations of deep learning

2023-10-20

* [Slides as PDF](/lecture02/pdf/nlpwdl2023-lecture02-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=viej0VnvwMY)

### Exercise 02: Computational graph and backpropagation from scratch

2023-10-26

* See the README.md and Python code under [exercises/ex02](exercises/ex02)

### Lecture 03: Text classification 1: Log-linear models

2023-10-27

* [Slides as PDF](/lecture03/pdf/nlpwdl2023-lecture03-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=BqvDHdOwCY4)
 
### Exercise 03: Gradient of a log-linear model, logistic loss, and parameter update

2023-11-02

* See the README.md and Python code under [exercises/ex03](exercises/ex03)

### Lecture 04: Text classification 2: Towards deep neural networks

2023-11-03

* [Slides as PDF](/lecture04/pdf/nlpwdl2023-lecture04-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=xOoJX_RXu50)

### Exercise 04: Linear layer, online gradient descent, training with data

2023-11-09

* See the README.md and Python code under [exercises/ex04](exercises/ex04)

### Lecture 05: Text generation 1: Language models and word embeddings

2023-11-10

* [Slides as PDF](/lecture05/pdf/nlpwdl2023-lecture05-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=NhL916qhdWs)

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

To install MS Segoe UI fonts required by the beamer template locally, run the following script (works also similarly on Mac OS): https://gist.github.com/habernal/ad1085ce5dc5e8cb3fbead354d8f4190

Run the script `compile-pdf.sh` in each lecture's folder to produce both handouts as well as unfolding PDFs used in the lecture.

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

Install fonts as above: https://gist.github.com/habernal/ad1085ce5dc5e8cb3fbead354d8f4190

Run the script `compile-pdf.sh` in each lecture's folder to produce both handouts as well as unfolding PDFs used in the lecture, which generates the PDFs in your local folder (e.g, `/tmp/slides`).

