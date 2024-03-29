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

### Exercise 05: ReLU and SoftMax

2023-11-16

* See the README.md and Python code under [exercises/ex05](exercises/ex05)

### Lecture 06: Text classification 3: Learning word embeddings

2023-11-17

* [Slides as PDF](/lecture06/pdf/nlpwdl2023-lecture06-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=3dVJX1vT4cw)

### Exercise 06: PyTorch 101

2023-11-23

* See the README.md and Python code under [exercises/ex06](exercises/ex06)

### Lecture 07: Text classification 4: Recurrent neural networks

2023-11-24

* [Slides as PDF](/lecture07/pdf/nlpwdl2023-lecture07-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=-83Fsl9csKA)

### Lecture 08: Text generation 2: Autoregressive encoder-decoder with RNNs and attention

2023-12-01

* [Slides as PDF](/lecture08/pdf/nlpwdl2023-lecture08-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=94bA72X8lDo)

### Lecture 09: Text classification 5: Introduction to transformers with BERT

2023-12-08

* [Slides as PDF](/lecture09/pdf/nlpwdl2023-lecture09-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=pTHlQXnr024)

### Exercise 09: Language modeling from scratch

2023-12-14

* See the README.md and Python code under [exercises/ex09](exercises/ex09)

### Lecture 10: Text classification 6: BERT part two

2023-12-15

* [Slides as PDF](/lecture10/pdf/nlpwdl2023-lecture10-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=squ4NM0sGTY)

### Lecture 11: Text generation 4: Decoder-only models and GPT

2023-12-22

* [Slides as PDF](/lecture11/pdf/nlpwdl2023-lecture11-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=clltK2E2xs8)


### Lecture 12: Text generation 5: Transformers and contemporary LLMs

2024-01-12

* [Slides as PDF](/lecture12/pdf/nlpwdl2023-lecture12-handout.pdf)
* [YouTube recording](https://www.youtube.com/watch?v=xOatNObLA0I)

### Lecture 13: "Guest lecture" on privacy in NLP

2024-01-19

* [YouTube recording](https://www.youtube.com/watch?v=KRVXpTlUt44)

### Lecture 14: Presenting and discussing homework solutions

* by Julius Broermann
	* https://github.com/JuliusBroermann/neural_machine_translation-nlp_with_deep_learning_course-paderborn_university


### Lecture 15: "Guest lecture" on ethics of generative AI (Dr. Thomas Arnold, TU Darmstadt)

2024-02-02

* [YouTube recording](https://www.youtube.com/watch?v=lO2-W5l2y40)



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

