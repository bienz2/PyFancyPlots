import plot

x = [1,4,3,6,8,8,2,4,2,4,7,8]
y = [[1,2,5,7,3,4,6,8,2,4,3,3],[3,4,5,6,3,4,5,6,6,5,4,3]]
labels = ["A", "B"]
plot.violin_plot(x, y, labels)
plot.display_plot()
