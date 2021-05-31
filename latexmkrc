# vim: set ft=perl:

@default_files = ('main.tex');

# xelatex
# http://blog.wxm.be/2015/08/05/use-xelatex-by-default-in-latexmk.html
$pdflatex = "xelatex %O %S";
$pdf_mode = 1;
$dvi_mode = $postscript_mode = 0;
