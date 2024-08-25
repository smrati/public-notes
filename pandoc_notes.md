# How to generate documents using `pandoc`


### what are the different syntax highligting styles available
```bash
pandoc --list-highlight-styles
```
1. pygments
2. tango
3. espresso
4. zenburn
5. kate
6. monochrome
7. breezedark
8. haddock

I like `breezedark`

---

### Useful commands
```bash
pandoc NER_in_spaCy.md -o NER_in_spaCy.pdf -V papersize:a4
pandoc -t beamer staging_train_custom_ner.md -o slides.pdf
```

---

How to control the font size
```
pandoc readme.md -o readme_article.pdf -V papersize:a4 -V fontsize=6pt --pdf-engine=xelatex --highlight-style=breezedark
```

---

How to export as presentation
```bash
pandoc -t beamer readme.md -o readme_article.pdf -V papersize:a4 -V fontsize=1pt --pdf-engine=xelatex --highlight-style=breezedark
```
