import jax.numpy as jnp

def ns_and_reward_partial(jax_compiled_model, s_keys, a_keys, ns_keys, s_gs_idx, a_ga_idx):

    reward_fn = jax_compiled_model.reward
    cpfs = jax_compiled_model.cpfs
    extra_params = jax_compiled_model.model_params
    levels = [_v for v in jax_compiled_model.levels.values() for _v in v]
    const_dict = {k:jax_compiled_model.init_values[k] for k in jax_compiled_model.rddl.nonfluents.keys()}

    del jax_compiled_model

    def _ns_and_reward(state, action, rng_key):
        """
        s_keys, a_keys: not grounded 
        gs_keys, ga_keys: grounded
        grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
        state, action: grounded
        """

        # s_gs_idx and s_ga_idx have 4 values for each key.
        # idx 0 and idx 1 are the min and max indexes for the key
        # idx 3 is the desired shape.
        state_dict = {k: jnp.array(state[s_gs_idx[k][0] : s_gs_idx[k][1]]).reshape(s_gs_idx[k][3]) for k in s_keys}
        action_dict = {k: jnp.array(action[a_ga_idx[k][0] : a_ga_idx[k][1]]).reshape(a_ga_idx[k][3]) for k in a_keys}

        # subs should be not grounded.
        subs = {**state_dict, **action_dict, **const_dict}

        for level in levels:
            expr = cpfs[level]
            subs[level], _, _ = expr(subs, extra_params, rng_key)
            
        reward, _, _ = reward_fn(subs, extra_params, rng_key)

        # flatten is required in cases like RecSim where state variables are 2D
        return jnp.hstack([subs[k].flatten() for k in ns_keys]), reward
    return _ns_and_reward

def ns_and_reward_partial_recsim(jax_compiled_model, s_keys, ns_keys, s_gs_idx, n_consumer, n_item):

    reward_fn = jax_compiled_model.reward
    cpfs = jax_compiled_model.cpfs
    extra_params = jax_compiled_model.model_params
    levels = [_v for v in jax_compiled_model.levels.values() for _v in v]
    const_dict = {k:jax_compiled_model.init_values[k] for k in jax_compiled_model.rddl.nonfluents.keys()}

    del jax_compiled_model

    def _ns_and_reward(state, action, rng_key):
        """
        s_keys, a_keys: not grounded 
        gs_keys, ga_keys: grounded
        grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
        state, action: grounded
        """

        # s_gs_idx and s_ga_idx have 4 values for each key.
        # idx 0 and idx 1 are the min and max indexes for the key
        # idx 3 is the desired shape.
        state_dict = {k: jnp.array(state[s_gs_idx[k][0] : s_gs_idx[k][1]]).reshape(s_gs_idx[k][3]) for k in s_keys}
        consumer = jnp.argmax(action)
        item = jnp.max(action[consumer]).astype(jnp.int32)
        ac_val = jnp.zeros((n_consumer, n_item))
        ac_val = ac_val.at[consumer, item].set(1)
        action_dict = {f"recommend": ac_val}

        # subs should be not grounded.
        subs = {**state_dict, **action_dict, **const_dict}

        for level in levels:
            expr = cpfs[level]
            subs[level], _, _ = expr(subs, extra_params, rng_key)
            
        reward, _, _ = reward_fn(subs, extra_params, rng_key)

        # flatten is required in cases like RecSim where state variables are 2D
        return jnp.hstack([subs[k].flatten() for k in ns_keys]), reward
    return _ns_and_reward