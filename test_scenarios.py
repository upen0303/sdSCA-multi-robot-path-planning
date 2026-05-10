"""
Test all 3 scenarios — verify configuration is correct.
Does NOT run full simulation (that comes in experiments/).
Just checks everything is set up properly.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.scenarios import (get_scenario,
                                     print_scenario_info)

print("=" * 55)
print("  Scenarios Configuration Test")
print("=" * 55)

for num in [1, 2, 3]:

    env = get_scenario(num)
    print_scenario_info(env, num)

    # Verify counts match paper
    expected = {
        1: (6,  7, 3, 100, 100),
        2: (7,  7, 3, 100, 100),
        3: (12, 14, 6, 200, 200),
    }

    exp = expected[num]
    assert len(env.robots)            == exp[0], \
           f"Wrong robot count!"
    assert len(env.static_obstacles)  == exp[1], \
           f"Wrong static obstacle count!"
    assert len(env.dynamic_obstacles) == exp[2], \
           f"Wrong dynamic obstacle count!"
    assert env.width                  == exp[3], \
           f"Wrong width!"
    assert env.height                 == exp[4], \
           f"Wrong height!"

    print(f"  ✅ Scenario {num} configuration correct!")

print("\n" + "=" * 55)
print("  ✅ All scenarios configured correctly!")
print("=" * 55)