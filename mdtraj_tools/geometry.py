import mdtraj as md
import numpy as np


def rmsd(trjs, ref, sel_idx, ref_idx):
    '''
    Calculate rmsd with respect to ref of selected atoms.

    Parameters
    ----------
    trjs : list or md.Trajectory or str
        List of mdtraj trajectories. Trajectories have to be stored as
        md.Trajectory or as basename, where the trajectory file is found under
        basename.xtc and the topology under basename.pdb.
    ref : str or None
        Path to pdb file with reference or None if reference is frame 0.
    sel_idx : np.ndarray
        Atom indices in trajectories. Either 1D array of size n_atoms or 2D of
        shape n_trajectories, n_atoms if atoms change indices.
    ref_idx : list or array-like
        Atom indices in ref file.

    Returns
    -------
    rmsds : np.ndarray
        RMSDs per frame of trajectory.
    '''
    if sel_idx.ndim == 1:
        if not len(sel_idx) == len(ref_idx):
            raise ValueError(
                f'sel_idx has length {len(sel_idx)}, ref_idx has length {len(ref_idx)}.')
    elif sel_idx.ndim == 2:
        if not sel_idx.shape[1] == len(ref_idx):
            raise ValueError(
                f'sel_idx must have shape (Any, {len(ref_idx)}), but has shape {sel_idx.shape}')
    else:
        raise ValueError('sel_idx has too many dimensions.')
    for trj in trjs:
        if not isinstance(trj, type(trj[0])):
            raise ValueError('All objects in trjs must be of same type.')

    rmsds = []

    if ref:
        t_ref = md.load(ref)

    for i_trj, trj in enumerate(trjs):
        t = _get_trj(trj)

        if sel_idx.ndim == 1:
            sel_traj = sel_idx
        else:
            sel_traj = sel_idx[i_trj]

        if ref:
            rms = md.rmsd(t, t_ref, atom_indices=sel_traj, ref_atom_indices=ref_idx)
        else:
            rms = md.rmsd(t, t, atom_indices=sel_traj, ref_atom_indices=ref_idx)

        rmsds.append(rms)
        del t

    rmsds = np.concatenate(rmsds)
    return rmsds


def dihedral(trjs, dihedrals):
    '''
    Compute dihedrals (phi, psi, chi1 or custom) for trajectory.

    Depending on the dihedrals parameter, calculate either all phi/psi/chi1
    dihedrals with built-in mdtraj functions or calculate dihedrals specified by
    a 2D array.

    Parameters
    ----------
    trjs : list or md.Trajectory or str
        List of mdtraj trajectories. Trajectories have to be stored as
        md.Trajectory or as basename, where the trajectory file is found under
        basename.xtc and the topology under basename.pdb.
    dihedrals : 'phi', 'psi', 'chi1' or np.ndarray
        Specifies which dihedrals are computed using the mdtraj functions.
        Custom dihedrals can be specified with a 2D array of shape n_dihedral, 4
        or with a 3D array of shape n_traj, n_dihedrals, 4.
    '''
    indices = []
    dihed_values = []

    if isinstance(dihedrals, np.ndarray):
        if dihedrals.ndim == 2:
            if not dihedrals.shape[1] == 4:
                raise ValueError(f'dihedrals is shape {dihedrals.shape}. Must be (any, 4).')

            for trj in trjs:
                t = _get_trj(trj)
                dh = md.compute_dihedrals(t, dihedrals)
                dihed_values.append(dh)

            indices = dihedrals
            dihed_values = np.concatenate(dihed_values)

            return (indices, dihed_values)

        elif dihedrals.ndim == 3:

            for i_trj, trj in enumerate(trjs):
                t = _get_trj(trj)
                dh = md.compute_dihedrals(t, dihedrals[i_trj])
                dihed_values.append(dh)

            indices = dihedrals
            dihed_values = np.concatenate(dihed_values)

            return (indices, dihed_values)

        else:
            raise ValueError('dihedrals must have 2 or 3 dimensions.')

    elif dihedrals == 'phi':
        pass
    elif dihedrals == 'psi':
        pass
    elif dihedrals == 'chi1':
        pass
    else:
        raise ValueError('Invalid dihedrals parameter.')


def _get_trj(trj):
    '''Load trj from basename if trj is of type str. Return trj if md.Trajectory.'''
    if isinstance(trj, str):
        t = md.load(trj + '.xtc', top=trj + '.pdb')
        return t
    elif isinstance(trj, md.Trajectory):
        return trj
    else:
        raise ValueError(f'trj is of type {type(trj)}. Must be str or md.Trajectory.')
