"""CPU contracts for deploying the trained scalar polynomial with Wan hooks."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import numpy as np
import torch

from eval.model.piecewise import PiecewisePolynomialMethod, lagrange_weights
from eval.pipeline import _generate_video, build_tasks
from eval.polynomial_prior import OnlineFitConfig


PACKAGE_ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=PACKAGE_ROOT.parent


def write_prior(path,steps=10):
    artifact={"kind":"piecewise_log_change_prior","version":1,
        "polynomial":{"break_steps":[1,steps-2],"coefficients_ascending":[[np.log(.02),0,0,0]]},
        "mirror_axis_step":(steps-1)/2,
        "online_fit":asdict(OnlineFitConfig(ridge_before_axis=.1,ridge_after_axis=.01,mirror_weight=.2)),
        "step_grid":list(range(1,steps-1)),
        "model_timestep_grid":[1000-10*s for s in range(1,steps-1)],
        "generation":{"task":"ti2v-5B","size":"1280*704","frame_num":121,
                      "sampling_steps":steps,"sample_solver":"unipc","sample_shift":5.,"guide_scale":5.}}
    path.write_text(json.dumps(artifact))
    return path


def run(method,teacher,steps=10):
    method.reset(steps,3,1)
    predictions={}
    for step in range(steps):
        # Deliberately large latent catches cancellation if a direct predictor
        # unnecessarily reconstructs its output through x+(v_hat-x).
        x=torch.full((2,2,3,3),1e9)
        for branch in (True,False):
            result=method.try_skip([x],torch.tensor([1000-10*step]))
            truth=torch.full_like(x,teacher(step,branch))
            if result is None:
                method.update([x],[truth])
            else:
                predictions[step,branch]=result[0].clone()
    return predictions


class PiecewiseDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=write_prior(self.root/'prior.json')

    def tearDown(self):
        self.temp.cleanup()

    def method(self,**kwargs):
        return PiecewisePolynomialMethod(str(self.path),**kwargs)

    def test_known_linear_outputs_and_cfg_isolation(self):
        teacher=lambda s,c: (1.+.13*s) if c else (-4.+.31*s)
        method=self.method(use_online_fit=False)
        pred=run(method,teacher)
        for (s,c),tensor in pred.items():
            torch.testing.assert_close(tensor,torch.full_like(tensor,teacher(s,c)),atol=2e-6,rtol=0)
        self.assertEqual(method.skipped_pairs,[3,4,7,8])
        self.assertEqual(method.calculated_pairs,[0,1,2,5,6,9])
        self.assertEqual([int(s) for s in method.trajectory.observed_steps],[1,2,6])
        self.assertEqual([n.step for n in method.nodes[True]],[6,9])
        self.assertEqual([n.step for n in method.nodes[False]],[6,9])

    def test_nonadjacent_real_nodes_recover_quadratic(self):
        teacher=lambda s,c: (1 if c else 2)+.2*s+.03*s*s
        method=self.method(use_online_fit=False,tensor_degree=2)
        pred=run(method,teacher)
        for (s,c),tensor in pred.items():
            torch.testing.assert_close(tensor,torch.full_like(tensor,teacher(s,c)),atol=8e-6,rtol=0)
        self.assertEqual(method.predictions[-1]['real_node_steps'],[2,5,6])
        w=lagrange_weights([0,.1,.8],1.2)
        self.assertAlmostEqual(float(w @ np.array([0,.1,.8])**2),1.2**2,places=12)

    def test_switches_change_prediction_without_changing_compute_budget(self):
        teacher=lambda s,c: (1+.02*s+.006*s*s+.0007*s**3)*(1 if c else 3)
        methods=[]; predictions=[]
        for fit,mirror,mode in ((False,False,'observed'),(True,False,'observed'),
                                (True,True,'observed'),(True,True,'anchor')):
            method=self.method(use_online_fit=fit,use_mirror_node=mirror,mirror_node_mode=mode)
            predictions.append(run(method,teacher)); methods.append(method)
        for method in methods:
            self.assertEqual(method.calculated_pairs,methods[0].calculated_pairs)
            self.assertEqual(method.skipped_pairs,methods[0].skipped_pairs)
            self.assertEqual(len(method.predictions),len(method.skipped_pairs))
            self.assertEqual([int(s) for s in method.trajectory.observed_steps],[1,2,6])
        self.assertGreater(float((predictions[0][8,True]-predictions[1][8,True]).abs().max()),1e-4)
        self.assertGreater(float((predictions[1][8,True]-predictions[2][8,True]).abs().max()),1e-5)
        self.assertGreater(float((predictions[2][8,True]-predictions[3][8,True]).abs().max()),1e-5)

    def test_reset_clears_video_state_and_skipped_nodes_cannot_be_observed(self):
        method=self.method()
        run(method,lambda s,c:1+.1*s)
        method.reset(10,3,1)
        self.assertFalse(method.trajectory.observed_steps)
        self.assertFalse(method.predictions)
        self.assertFalse(method.nodes[True])
        x=torch.ones(2,2,3,3)
        for step in range(3):
            for _ in range(2):
                self.assertIsNone(method.try_skip([x],1000-10*step))
                method.update([x],[x*(step+1)])
        with self.assertRaisesRegex(RuntimeError,'scheduled real'):
            method.update([x],[x])

    def test_invalid_switches_schedule_and_generation_fail_early(self):
        with self.assertRaises(ValueError): self.method(use_online_fit=False,use_mirror_node=True)
        with self.assertRaises(ValueError): self.method(full_block_steps=1)
        with self.assertRaises(ValueError): self.method(tensor_degree=3)
        method=self.method()
        with self.assertRaises(ValueError): method.reset(50,3,1)
        with self.assertRaises(ValueError): method.reset(10,2,1)
        with self.assertRaises(ValueError): method.reset(10,3,0)
        method.reset(10,3,1)
        x=torch.ones(1)
        for _ in range(2):
            method.try_skip([x],1000); method.update([x],[x])
        with self.assertRaisesRegex(ValueError,'timestep'):
            method.try_skip([x],991)
        cfg=OmegaConf.create(dict(task='ti2v-5B',size='1280*704',frame_num=121,
            sample_steps=10,sample_solver='unipc',sample_shift=5.,sample_guide_scale=5.,image=None))
        method.validate_generation(cfg)
        cfg.sample_shift=3.
        with self.assertRaisesRegex(ValueError,'sample_shift'):
            method.validate_generation(cfg)

    def test_main_sweep_instantiates_piecewise_with_all_switches_and_prompts(self):
        prompt_file=self.root/'prompts.txt'
        prompt_file.write_text('\n'.join(f'prompt {i}' for i in range(7)))
        with initialize_config_dir(version_base=None,config_dir=str(PACKAGE_ROOT/'conf')):
            cfg=compose(config_name='sweep')
        cfg.prompts.file=str(prompt_file)
        entry=next(m for m in cfg.methods if m.name=='piecewise')
        schedules=[]
        for fit,mirror,mode in ((False,False,'observed'),(True,False,'observed'),
                                (True,True,'observed'),(True,True,'anchor')):
            entry.method.use_online_fit=fit
            entry.method.use_mirror_node=mirror
            with open_dict(entry.method):
                entry.method.mirror_node_mode=mode
            tasks=build_tasks(cfg,PROJECT_ROOT)
            self.assertEqual(len(tasks),91)
            piecewise_tasks=[t for t in tasks if t.method=='piecewise']
            self.assertEqual([t.prompt_id for t in piecewise_tasks],[str(i) for i in range(7)])
            task=piecewise_tasks[0]
            self.assertIsNone(task.cache_threshold)
            self.assertEqual(task.output_group,'piecewise')
            method=instantiate(task.method_config)
            self.assertIsInstance(method,PiecewisePolynomialMethod)
            self.assertEqual(method.options,dict(use_online_fit=fit,use_mirror_node=mirror,mirror_node_mode=mode))
            method.reset(int(cfg.generation.sample_steps),task.warmup_steps,task.final_full_steps)
            method.validate_generation(cfg.generation)
            schedules.append(method.summary()['planned_full_pairs'])
        self.assertEqual(len(schedules[0]),28)
        self.assertTrue(all(s==schedules[0] for s in schedules))

    def test_real_generation_hook_uses_piecewise_method_and_restores_forward(self):
        with initialize_config_dir(version_base=None,config_dir=str(PACKAGE_ROOT/'conf')):
            cfg=compose(config_name='sweep')
        # Keep only the method under test, using the normal sweep entry.
        cfg.methods=[m for m in cfg.methods if m.name=='piecewise']
        cfg.paths.piecewise_artifact=str(self.path)
        cfg.generation.sample_steps=10
        cfg.generation.warmup_steps=3
        cfg.methods[0].method.use_online_fit=True
        cfg.methods[0].method.use_mirror_node=True
        task=next(t for t in build_tasks(cfg,PROJECT_ROOT) if t.method=='piecewise')

        class FakeDiT(torch.nn.Module):
            def forward(self,x,t,context,seq_len,**kwargs):
                step=(1000-float(t[0]))/10
                return [value*.1+(1 if context[0] else 2)+.03*step for value in x]

        model=FakeDiT()
        original=model.forward
        states=[]
        def generate(*args,**kwargs):
            x=torch.ones(2,2,3,3)
            for step in range(10):
                t=torch.tensor([1000.-10*step])
                c=model([x],t,[True],1)[0]
                u=model([x],t,[False],1)[0]
                x=x-.01*(u+5*(c-u))
                states.append(x.clone())
            return x
        runtime=SimpleNamespace(pipeline=SimpleNamespace(model=model,generate=generate),
            size_configs={str(cfg.generation.size):(1,1)},max_area_configs={str(cfg.generation.size):1},
            wan_config=SimpleNamespace(sample_fps=16),
            save_video=lambda **kw:Path(kw['save_file']).write_bytes(b'synthetic hook test, not video'))
        with (patch('eval.pipeline.torch.cuda.synchronize'),
              patch('eval.pipeline.torch.cuda.is_available',return_value=False)):
            result=_generate_video(task,cfg,runtime,PROJECT_ROOT,self.root/'video.mp4',self.root/'log.txt')
        self.assertEqual(model.forward,original)
        self.assertEqual(result['timing']['dit_forward_calls'],12)
        self.assertEqual(result['timing']['cache_forward_calls'],8)
        self.assertTrue(result['cache']['use_online_fit'])
        self.assertTrue(result['cache']['use_mirror_node'])
        self.assertEqual([r['step'] for r in result['cache']['real_scalar_nodes']],[1,2,6])
        self.assertTrue(all(torch.isfinite(x).all() for x in states))
        json.dumps(result)

    def test_exported_50_step_prior_runs_all_ablations_in_a_synthetic_loop(self):
        artifact=PACKAGE_ROOT/'artifacts/piecewise_polynomial/prior.json'
        for fit,mirror,mode in ((False,False,'observed'),(True,False,'observed'),
                                (True,True,'observed'),(True,True,'anchor')):
            method=PiecewisePolynomialMethod(str(artifact),use_online_fit=fit,
                use_mirror_node=mirror,mirror_node_mode=mode)
            method.reset(50,7,1)
            x=torch.ones(2,2,3,3)
            for step in range(50):
                timestep=method.expected_timesteps.get(step,999 if step==0 else 92)
                pair=[]
                for branch in (True,False):
                    value=method.try_skip([x],timestep)
                    if value is None:
                        v=.01*x+(1 if branch else 1.4)+.03*step+.0005*step**2
                        method.update([x],[v])
                    else:
                        v=value[0]
                    pair.append(v)
                x=x-.002*(pair[1]+5*(pair[0]-pair[1]))
                self.assertTrue(torch.isfinite(x).all())
            self.assertEqual(len(method.calculated_pairs),28)
            self.assertEqual(len(method.skipped_pairs),22)
            self.assertEqual(len(method.predictions),22)
            self.assertEqual(method.trajectory.observed_steps,[1.,2.,3.,4.,5.,6.,10.,14.,18.,22.,26.,30.,34.,38.,42.,46.])
            self.assertEqual(method.predictions[-1]['step'],48)
            json.dumps(method.summary())


if __name__=='__main__':
    unittest.main()
