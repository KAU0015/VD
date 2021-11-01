set terminal pdfcairo enhanced color font "sans,6" fontscale 1.0 linewidth 1 rounded dashlength 1 background "white" size 10cm,6cm 
set encoding utf8
set samples 100

set output "cv1.pdf"

#definovani barvy os
set style line 80 linetype 1 linecolor "#808080" linewidth 2 
set border 3 back linestyle 80



set xtics nomirror
set ytics nomirror

set title "January Temperature Extremes" font "sans-Bold"
set xlabel "Time [day]"
set ylabel "Temperature [°C]"

#formatovani data
#set xdata time
#set timefmt "%d.%m.%Y"
#set format x "%d.%m.%Y"
#set xtics rotate by -90

#set xrange ["01.01.202100":"10.01.2021"]

#set key outside bottom right Left title "Probes:" enhanced font "sans-Italic" #reverse box

set xrange [1:31]
#set yrange [0:50]

#set style line 1 lt rgb "#A00000" lw 2 pt 1

#plot "data_temp.txt" index 0 title "Temperatures" w lp ls 1, \
#"" index 1 title "Inside" with lines

f(x) = a + b*x + c*x**2 + d*x**3
fit f(x) "data_temp.txt" using 1:2 via a,b,c,d

plot "data_temp.txt" index 0 title "Temperature" w lp ls 1, \
"" index 1 title "Temperature" with lines, f(x) title "Temperature approx."