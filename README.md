# Modified Adsorption: Python Implementation

This project implements P. Talukdar's Modified Adsorption (MAD) label propagation algorithm.
MAD is a graph semi-supervised learning model. As such it uses a small number of seeds to determine the class. It is transductive, it assigns classes to the already existent dataset. It does not assign classes to new, unseen instances.

## Project Setup

The module includes a simple class implementing the MAD algorithm. A graph file is needed, _graph\_file_, and a _seed\_file_, containing the minimum set of initial labeled items. They must be in a tab separated format: NodeA, NodeB, LinkWeight. For example, the graph_file should be like this:

    N1	N2	0.18

For the seed\_file, a Node, a Label and a Weight (indicating the strength of the label class):

    N1	L1	1.0
    N4	L2	1.0

Then, it suffices to call the MAD constructor:

    mad = ModifiedAdsorption(graph_file, seed_file)

Calculate the modified adsorption:

    mad.calculate_mad()

And finally, get the results:

    mad.results()

## Notes

The code is not thoroughly tested, the matrices are not checked for special considerations neither at the beginning of the process, neither at the iterative steps.

I am sure it could be made faster and more memory efficient. I have tested it with around 10k nodes and it took around 1hr to converge. I need to perform more experiments, specially take time to run these datasets: http://www.talukdar.net/datasets/class_inst/

## References

 Partha Pratim Talukdar and Koby Crammer. 2009. _New Regularized Algorithms for Transductive Learning._ In Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases: Part II (ECML PKDD '09), Wray Buntine, Marko Grobelnik, Dunja Mladenić, and John Shawe-Taylor (Eds.). Springer-Verlag, Berlin, Heidelberg, 442-457. DOI=10.1007/978-3-642-04174-7_29 http://dx.doi.org/10.1007/978-3-642-04174-7\_29


## License
Apache v2
