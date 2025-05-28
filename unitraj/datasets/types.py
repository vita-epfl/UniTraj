from metadrive.scenario.scenario_description import MetaDriveType

object_type = {
    MetaDriveType.UNSET: 0,
    MetaDriveType.VEHICLE: 1,
    MetaDriveType.PEDESTRIAN: 2,
    MetaDriveType.CYCLIST: 3,
    MetaDriveType.OTHER: 4,
}

# lane_type = {
#     0: MetaDriveType.LANE_UNKNOWN,
#     1: MetaDriveType.LANE_FREEWAY,
#     2: MetaDriveType.LANE_SURFACE_STREET,
#     3: MetaDriveType.LANE_BIKE_LANE
# }
#
# road_line_type = {
#     0: MetaDriveType.LINE_UNKNOWN,
#     1: MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
#     2: MetaDriveType.LINE_SOLID_SINGLE_WHITE,
#     3: MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
#     4: MetaDriveType.LINE_BROKEN_SINGLE_YELLOW,
#     5: MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW,
#     6: MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
#     7: MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
#     8: MetaDriveType.LINE_PASSING_DOUBLE_YELLOW
# }
#
# road_edge_type = {
#     0: MetaDriveType.LINE_UNKNOWN,
#     # // Physical road boundary that doesn't have traffic on the other side (e.g.,
#     # // a curb or the k-rail on the right side of a freeway).
#     1: MetaDriveType.BOUNDARY_LINE,
#     # // Physical road boundary that separates the car from other traffic
#     # // (e.g. a k-rail or an island).
#     2: MetaDriveType.BOUNDARY_MEDIAN
# }

polyline_type = {
    # for lane
    MetaDriveType.LANE_FREEWAY: 1,
    MetaDriveType.LANE_SURFACE_STREET: 2,
    'LANE_SURFACE_UNSTRUCTURE': 2,
    MetaDriveType.LANE_BIKE_LANE: 3,
    # for roadline
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 6,
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: 7,
    'ROAD_EDGE_SIDEWALK': 7,
    MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 8,
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 9,
    MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 10,
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 11,
    MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 12,
    MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 13,

    # for roadedge
    MetaDriveType.BOUNDARY_LINE: 15,
    MetaDriveType.BOUNDARY_MEDIAN: 16,

    # for stopsign
    MetaDriveType.STOP_SIGN: 17,

    # for crosswalk
    MetaDriveType.CROSSWALK: 18,

    # for speed bump
    MetaDriveType.SPEED_BUMP: 19,

}

traffic_light_state_to_int = {
    None: 0,
    MetaDriveType.LANE_STATE_UNKNOWN: 0,

    # // States for traffic signals with arrows.
    MetaDriveType.LANE_STATE_ARROW_STOP: 1,
    MetaDriveType.LANE_STATE_ARROW_CAUTION: 2,
    MetaDriveType.LANE_STATE_ARROW_GO: 3,

    # // Standard round traffic signals.
    MetaDriveType.LANE_STATE_STOP: 4,
    MetaDriveType.LANE_STATE_CAUTION: 5,
    MetaDriveType.LANE_STATE_GO: 6,

    # // Flashing light signals.
    MetaDriveType.LANE_STATE_FLASHING_STOP: 7,
    MetaDriveType.LANE_STATE_FLASHING_CAUTION: 8,
}

smart_traffic_light_state_to_int = {
    None: 0,
    MetaDriveType.LANE_STATE_UNKNOWN: 3,

    MetaDriveType.LANE_STATE_ARROW_STOP: 0,
    MetaDriveType.LANE_STATE_STOP: 0,
    MetaDriveType.LANE_STATE_FLASHING_STOP: 0,

    MetaDriveType.LANE_STATE_ARROW_CAUTION: 2,
    MetaDriveType.LANE_STATE_CAUTION: 2,
    MetaDriveType.LANE_STATE_FLASHING_CAUTION: 2,

    MetaDriveType.LANE_STATE_ARROW_GO: 1,
    MetaDriveType.LANE_STATE_GO: 1,
}

smart_polygon_type = {
    'VEHICLE':0,
    'BIKE': 1,
    'BUS': 2,
    'PEDESTRIAN': 3,
    'UNKNOWN': 4,
}