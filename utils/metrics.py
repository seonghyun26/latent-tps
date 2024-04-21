import torch
import numpy as np

# Basic calculations

def compute_dihedral(p): 
    """http://stackoverflow.com/q/20305272/1128289"""
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)

def get_log_normal(x):
    normal = torch.distributions.normal.Normal(loc=0, scale=1)
    return normal.log_prob(x)

def get_dist_matrix(x):
    x = x.view(x.shape[0], -1, 3)
    dist_matrix = torch.cdist(x, x)
    return dist_matrix


# Metrics

def expected_pairwise_distance(last_position, target_position):
    last_dist_matrix = get_dist_matrix(last_position)
    target_dist_matrix = get_dist_matrix(target_position)
    
    epd = torch.mean((last_dist_matrix-target_dist_matrix)**2).item()
    return 1000*epd

def target_hit_percentage(last_position, target_position):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, 0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, 0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, 0, angle_1, :])
        phi = compute_dihedral(last_position[i, 0, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            hit += 1
    
    thp = int(100*hit/last_position.shape[0])
    return thp

def energy_transition_point(last_position, target_position, potentials):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    etp = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, 0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, 0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, 0, angle_1, :])
        phi = compute_dihedral(last_position[i, 0, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            etp += potentials[i].max()
            hit += 1
    
    etp = etp.item() / hit if hit > 0 else None
    return etp