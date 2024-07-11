import numpy as np
from biorbd_casadi import NodeSegment
from casadi import Function

from wheelchair_utils.custom_biorbd_model_holonomic import BiorbdModelCustomHolonomic

biorbd_model_path = "models/wheelchair_model.bioMod"

bio_model = (
    BiorbdModelCustomHolonomic(biorbd_model_path, "push_phase"),
    BiorbdModelCustomHolonomic(biorbd_model_path, "recovery_phase"),
)
node = NodeSegment(1, 3, 4)
bio_model[0].model.setMarker(index=0, pos=node)
output = bio_model[0].markers(np.zeros(4))[0]
func = Function("a", [], [output])
print(func())
