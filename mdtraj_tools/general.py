import numpy as np
import mdtraj as md
from .geometry import _get_trj


def trj_data(trjs):
    '''
    Save general trajectory data.

    Parameters
    ----------
    trjs : list of md.Trajectory or list of str
        MDTraj Trajectories.
    '''
    columns = ['n_chains',
               'n_residues',
               'n_atoms',
               'n_protein_residues',
               'n_protein_atoms',
               'n_water_residues',
               'n_water_atoms',
               'n_ion_atoms',
               'box_vector_x',
               'box_vector_y',
               'box_vector_z']
    data = np.zeros(len(trjs), len(columns))
