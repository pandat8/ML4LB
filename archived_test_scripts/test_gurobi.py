import gurobipy as gp

try:
    # Attempt to initialize a blank model environment
    with gp.Env() as env, gp.Model(env=env) as model:
        print("\n✅ Success! Gurobi license is active and working.")
except gp.GurobiError as e:
    print(f"\n❌ Gurobi Error: {e}")
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")