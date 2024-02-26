"""
This file is to display the human model into bioviz
"""
import bioviz

model_name = "wheelchair_model.bioMod"

biorbd_viz = bioviz.Viz(model_name,
    show_gravity_vector=False,
    show_floor=False,
    show_local_ref_frame=True,
    show_global_ref_frame=True,
    show_markers=True,
    show_mass_center=False,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=True,
    mesh_opacity=0.5,
    background_color=(0.5, 0.5, 0.5),
                         )

biorbd_viz.exec()
print("Done")
