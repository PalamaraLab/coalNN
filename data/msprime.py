import msprime
import numpy as np
import tskit


def simulate(sample_size, config, seed, rec_map, Ne, sample_dict=None):
    """
    https://msprime.readthedocs.io/en/stable/
    """
    if sample_dict is not None:
        samples = get_samplesets(sample_dict)
    else:
        samples = sample_size

    if config.demography != 'constant':
        demo_path = './files/demography/' + str(config.demography) + '.popsizes_1kg.generations.mu1.65E-8.demo'
        # print("Simulating demographic model from " + demo_path + "...")
        generations, pop_sizes, growth_rates = read_demography(demo_path)
        demography = get_demography(generations, pop_sizes, growth_rates)
        population_size = None

    else:
        # print("No demographic model is specified...")
        demography = None
        population_size = Ne

    if hasattr(config, 'rec_rate') and hasattr(config, 'length'):
        if hasattr(config, 'gc_rate') and config.gc_rate != 0:
            ts = msprime.sim_ancestry(random_seed=seed,
                                  population_size=population_size,
                                  demography=demography,
                                  recombination_rate=config.rec_rate,
                                  sequence_length=config.length,
                                  gene_conversion_rate=config.gc_rate,
                                  gene_conversion_tract_length=300, 
                                  samples=samples,
                                  ploidy=2)
        else:
            ts = msprime.sim_ancestry(random_seed=seed,
                                  population_size=population_size,
                                  demography=demography,
                                  recombination_rate=config.rec_rate,
                                  sequence_length=config.length,
                                  samples=samples,
                                  ploidy=2)
    else:
        rate_map = msprime.RateMap.read_hapmap(rec_map)
        ts = msprime.sim_ancestry(random_seed=seed,
                                  population_size=population_size,
                                  demography=demography,
                                  recombination_rate=rate_map,
                                  samples=samples,
                                  ploidy=2)
    mts = msprime.sim_mutations(ts,
                                rate=config.muration_rate,
                                random_seed=seed,
                                discrete_genome=False)

    return mts

# for msprime 1.0
def get_demography(generations, pop_sizes, growth_rates):
    demography = msprime.Demography()
    demography.add_population(initial_size=pop_sizes[0],
                              growth_rate=growth_rates[0])
    n_demo_events = len(pop_sizes)
    for i in range(1, n_demo_events):
        demography.add_population_parameters_change(time=generations[i],
                                                    initial_size=pop_sizes[i],
                                                    growth_rate=growth_rates[i])
    return demography


def read_demography(file):
    with open(file) as f:
        lines = f.readlines()
        generations = []
        pop_sizes = []
        growth_rates = []

        for line in lines:
            line_split = line.split()
            gen = int(float(line_split[0]))
            # Population sizes correspond to diploid
            pop_size = float(line_split[1])

            generations.append(gen)
            pop_sizes.append(pop_size)

            if len(pop_sizes) > 1:
                growth_rates.append(
                    np.log(pop_sizes[-2] / pop_sizes[-1]) * 1 / (generations[-1] - generations[-2])
                )

        growth_rates.append(0)

    return generations, pop_sizes, growth_rates


def get_samplesets(sample_dict):
    samples = []
    for key, val in sample_dict.items():
        if "anc" not in key:
            samples += [msprime.SampleSet(val[0], time=val[1])]
        # Ancient samples from population POP_NAME should have key value POP_NAME_anc
        else:
            samples += [msprime.SampleSet(val[0], time=val[1])]
    return (samples)


def save_simulation(simulation, filename):
    tskit.TreeSequence.dump(simulation, filename)
    return


def restore_simulation(filename):
    return tskit.TreeSequence.load(filename)
