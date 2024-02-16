"""
Here is the code that helped me to define the parameters of the model.
Including the wheelchair and the subject.
"""

# toutes les unités sont en S.I. (mètres)

# segment 1
length_arm = 0.271  # m  # Epaule au coude
# segment 2
length_forearm = 0.283  # m  # Coude au poignet
length_hand = 0.10  # m  # poignet au milieu des deux tetes du metacarpes 2 et 5.

# On évaluera la masse des membres à partir de la masse totale du sujet
# masse totale du sujet
# CONSIGNE: entrez votre masse
m_tot = 80  # kg

# (selon Dumas et al.,2007) Adjustments to McConville et al. and Young et al. body segment inertial parameters
m_arm = 2 * 0.024 * m_tot  # kg
m_forearm = 2 * 0.017 * m_tot  # kg
m_hand = 2 * 0.006 * m_tot  # kg

m_pelvis = 0.142 * m_tot  # kg
m_head = 0.067 * m_tot
m_torso = 0.333 * m_tot

m_thigh = 2 * 0.123 * m_tot  # kg
m_shank = 2 * 0.048 * m_tot  # kg
m_foot = 2 * 0.012 * m_tot  # kg

length_head = 244 * 10 ** -3  # m
length_torso = 477 * 10 ** -3  # m
length_pelvis = 94 * 10 ** -3  # m

length_thigh = 432 * 10 ** -3  # m
length_shank = 433 * 10 ** -3  # m
length_foot = 165 * 10 ** -3  # m

# on donne ici des valeurs numériques des différents paramètres du modèle
# Upper arm
p = {}
p['l1'] = length_arm  # m
p['m1'] = m_arm  # masse segment 1 en kg
p['c1'] = p['l1'] * 0.5502  # position centre de masse segment 1 en m  # (selon Dumas et al.,2007)
p['I1'] = (0.33 * length_arm) ** 2 * m_arm  # kg.m2  # (selon Dumas et al.,2007)

# Forearm + hand + glass
I_Forearm = (0.27 * length_forearm) ** 2 * m_forearm  # kg.m2  # (selon Dumas et al.,2007)
I_Hand = (0.56 * length_hand) ** 2 * m_hand  # kg.m2  # (selon Dumas et al.,2007)

p['m2'] = m_forearm + m_hand
p['l2'] = length_forearm + length_hand  # m
p['c2'] = (m_forearm * length_forearm * 0.5726 +
           m_hand * (length_forearm + length_hand * 0.6309)) / (
                      m_forearm + m_hand)  # position centre de masse segment 2 en m
p['I2'] = I_Forearm + m_forearm * (p['c2'] - length_forearm * 0.572) ** 2 \
          + I_Hand + m_hand * (p['c2'] - (length_forearm + length_hand * 0.6309)) ** 2

# Defintion Roue (wheel)
# FRET 1
p['m_w'] = 3.79  # Kg  these C.sauret
p['I_w'] = 0.160  # kg.m² these C.sauret
# position du centre de masse de la roue dans son repère locale
p['c_wx'] = 0
p['c_wy'] = 0
# radius of the handrim / rayon de la main courante
p['R_hr'] = 0.3
# radius of the wheel / rayon de la roue
p['R_w'] = 0.35

# DEFINITION DU Sujet - FAUTEUIL
# Origine en y du fauteuil dans R0
p['c_wy'] = p['R_w']
# masse fauteuil
m_mwc = 17  # kg; % These samuel Hybois

# masse fauteuil + sujet
p['m_mwc'] = m_pelvis + m_head + m_torso + m_thigh + m_shank + m_foot + m_mwc

# hauteur de l'assise; Hybois % selon y;
p['hs'] = 0.2  # Rfauteuil

# position du fauteuil et du sujet dans le repère fauteuil
# centré sur la roue du fauteuil
c_mwcx = 0.166  # fret 1 sauret;
c_mwcy = 0.089  # fret 1 sauret;

# Y verticale
p['c_mwcy'] = ((p['hs'] + length_pelvis * (1 - 0.28)) * m_pelvis +
               (p['hs'] + length_pelvis + length_torso * (1 - 0.420)) * m_torso +
               (p['hs'] + length_pelvis + length_torso + length_head * 0.555) * m_head +
               p['hs'] * m_thigh +
               (p['hs'] - length_shank * 0.404) * m_shank +
               (p['hs'] - length_shank) * m_foot +
               c_mwcy * m_mwc) / p['m_mwc']
# X horizontale
p['c_mwcx'] = (0 * m_pelvis +
               0 * m_torso +
               0 * m_head +
               length_thigh * 0.429 * m_thigh +
               length_thigh * m_shank +
               (length_thigh + length_foot * 0.436) * m_foot +
               c_mwcy * m_mwc) / p['m_mwc']

# position de l'épaule
p['shy'] = p['hs'] + length_pelvis + length_torso

print("Done!")
# print everything in the dict p
print(p)
