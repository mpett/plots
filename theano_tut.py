import theano
from theano import tensor as T

x=T.vector('x')
W=T.matrix('W')
b=T.vector('b')
dot=T.dot(x,W)
out=T.nnet.sigmoid(dot+b)
from theano.printing import debugprint
debugprint(dot)
debugprint(out)
f=theano.function(inputs=[x,W],outputs=dot)
g=theano.function([x,W,b],out)
h=theano.function([x,W,b],[dot,out])
i=theano.function([x,W,b],[dot+b,out])
debugprint(f)
debugprint(g)
from theano.printing import pydotprint
pydotprint(f,outfile='pydotprint_f.png')
from IPython.display import Image
Image('pydotprint_f.png',width=1000)

