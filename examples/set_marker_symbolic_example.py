"""
Example of setting a marker on a segment in a biorbd model
and retrieving its position symbolically with CasADi.
"""


import numpy as np
from biorbd_casadi import NodeSegment
from casadi import Function

from wheelchair_utils.custom_biorbd_model_holonomic_new import HolonomicTorqueWheelchairModel

biorbd_model_path = "models/wheelchair_model.bioMod"

bio_model = (
    HolonomicTorqueWheelchairModel(biorbd_model_path, "push_phase"),
    HolonomicTorqueWheelchairModel(biorbd_model_path, "recovery_phase"),
)
node = NodeSegment(1, 3.35, 4)
bio_model[0].model.setMarker(index=0, pos=node)
output = bio_model[0].model.markers(np.zeros(4))[0].to_mx()
func = Function("a", [], [output])
print(func())
