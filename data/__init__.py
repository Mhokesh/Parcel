"""Problem datasets and generators."""
def __getattr__(name):
    if name in ("CVRPInstanceGenerator", "sample_cvrp_batch"):
        from data import cvrp_generator
        return getattr(cvrp_generator, name)
    if name in ("load_cvrplib_benchmark", "parse_vrp_string"):
        from data import benchmark_parser
        return getattr(benchmark_parser, name)
    raise AttributeError(name)
