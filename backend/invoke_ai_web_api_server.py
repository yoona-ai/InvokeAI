import os
import traceback

from flask import Flask, redirect, send_from_directory, request as flask_request
from uuid import uuid4

from ldm.invoke.args import Args, APP_ID, APP_VERSION, calculate_init_img_hash
from ldm.invoke.pngwriter import PngWriter, retrieve_metadata
from ldm.invoke.conditioning import split_weighted_subprompts

from backend.modules.parameters import parameters_to_command
from backend.storage_adapter import S3Adapter


# Loading Arguments
opt = Args()
args = opt.parse_args()


class InvokeAIWebAPIServer:
    def __init__(self, generate, gfpgan, codeformer, esrgan) -> None:
        self.app = None
        self.host = args.host
        self.port = args.port

        self.generate = generate
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan

        self.export_path = None

    def run(self):
        self.setup_app()
        self.setup_flask()

    def setup_flask(self):
        # Socket IO
        self.app = Flask(__name__)

        # Base Route
        @self.app.route('/images/', methods=['POST'])
        def serve():
            # TODO JHILL: minimal error checking of supplied data?
            # TODO JHILL: what if no prompt?

            generation_parameters = {
                'steps': int(flask_request.json.get('steps', 1)),
                'prompt': flask_request.json.get('prompt'),
                'sampler_name': 'k_lms'
            }

            esrgan_parameters = {

            }

            gfpgan_parameters = {

            }

            path = self.generate_images(generation_parameters, esrgan_parameters, gfpgan_parameters)
            image_url = S3Adapter().store_from_local_file(path, user_id=flask_request.json['user_id'])

            return {
                'image_url': image_url
            }

        self.app.run()

    def setup_app(self):
        self.result_url = 'outputs/'
        self.init_image_url = 'outputs/init-images/'
        self.mask_image_url = 'outputs/mask-images/'
        self.intermediate_url = 'outputs/intermediates/'
        # location for "finished" images
        self.result_path = args.outdir
        # temporary path for intermediates
        self.intermediate_path = os.path.join(
            self.result_path, 'intermediates/'
        )
        # path for user-uploaded init images and masks
        self.init_image_path = os.path.join(self.result_path, 'init-images/')
        self.mask_image_path = os.path.join(self.result_path, 'mask-images/')
        # txt log
        self.log_path = os.path.join(self.result_path, 'invoke_log.txt')
        # make all output paths
        [
            os.makedirs(path, exist_ok=True)
            for path in [
                self.result_path,
                self.intermediate_path,
                self.init_image_path,
                self.mask_image_path,
            ]
        ]

    # App Functions
    def get_system_config(self):
        return {
            'model': 'stable diffusion',
            'model_id': args.model,
            'model_hash': self.generate.model_hash,
            'app_id': APP_ID,
            'app_version': APP_VERSION,
        }

    def generate_images(self, generation_parameters, esrgan_parameters, gfpgan_parameters):
        try:
            step_index = 1
            prior_variations = (
                generation_parameters['with_variations'] if 'with_variations' in generation_parameters else []
            )

            # We need to give absolute paths to the generator, stash the URLs for later
            init_img_url = None
            mask_img_url = None

            if 'init_img' in generation_parameters:
                init_img_url = generation_parameters['init_img']
                generation_parameters[
                    'init_img'
                ] = self.get_image_path_from_url(
                    generation_parameters['init_img']
                )

            if 'init_mask' in generation_parameters:
                mask_img_url = generation_parameters['init_mask']
                generation_parameters[
                    'init_mask'
                ] = self.get_image_path_from_url(
                    generation_parameters['init_mask']
                )

            def image_done(image, seed, first_seed):
                nonlocal generation_parameters
                nonlocal esrgan_parameters
                nonlocal gfpgan_parameters
                nonlocal prior_variations

                all_parameters = generation_parameters
                postprocessing = False

                if (
                    'variation_amount' in all_parameters
                    and all_parameters['variation_amount'] > 0
                ):
                    first_seed = first_seed or seed
                    this_variation = [
                        [seed, all_parameters['variation_amount']]
                    ]
                    all_parameters['with_variations'] = (
                        prior_variations + this_variation
                    )
                    all_parameters['seed'] = first_seed
                elif 'with_variations' in all_parameters:
                    all_parameters['seed'] = first_seed
                else:
                    all_parameters['seed'] = seed

                if esrgan_parameters:
                    image = self.esrgan.process(
                        image=image,
                        upsampler_scale=esrgan_parameters['level'],
                        strength=esrgan_parameters['strength'],
                        seed=seed,
                    )

                    postprocessing = True
                    all_parameters['upscale'] = [
                        esrgan_parameters['level'],
                        esrgan_parameters['strength'],
                    ]

                if gfpgan_parameters:
                    image = self.gfpgan.process(
                        image=image,
                        strength=gfpgan_parameters['strength'],
                        seed=seed,
                    )
                    postprocessing = True
                    all_parameters['facetool_strength'] = gfpgan_parameters[
                        'strength'
                    ]

                # restore the stashed URLS and discard the paths, we are about to send the result to client
                if 'init_img' in all_parameters:
                    all_parameters['init_img'] = init_img_url

                if 'init_mask' in all_parameters:
                    all_parameters['init_mask'] = mask_img_url

                metadata = self.parameters_to_generated_image_metadata(
                    all_parameters
                )

                command = parameters_to_command(all_parameters)

                path = self.save_result_image(
                    image,
                    command,
                    metadata,
                    self.result_path,
                    postprocessing=postprocessing,
                )

                print(f'>> Image generated: "{path}"')
                self.write_log_message(f'[Generated] "{path}": {command}')
                self.export_path = path

            self.generate.prompt2image(
                **generation_parameters,
                image_callback=image_done,
            )

            return self.export_path

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(e)
            print('\n')

            traceback.print_exc()
            print('\n')

    def parameters_to_generated_image_metadata(self, parameters):
        try:
            # top-level metadata minus `image` or `images`
            metadata = self.get_system_config()
            # remove any image keys not mentioned in RFC #266
            rfc266_img_fields = [
                'type',
                'postprocessing',
                'sampler',
                'prompt',
                'seed',
                'variations',
                'steps',
                'cfg_scale',
                'threshold',
                'perlin',
                'step_number',
                'width',
                'height',
                'extra',
                'seamless',
                'hires_fix',
            ]

            rfc_dict = {}

            for item in parameters.items():
                key, value = item
                if key in rfc266_img_fields:
                    rfc_dict[key] = value

            postprocessing = []

            # 'postprocessing' is either null or an
            if 'facetool_strength' in parameters:

                postprocessing.append(
                    {
                        'type': 'gfpgan',
                        'strength': float(parameters['facetool_strength']),
                    }
                )

            if 'upscale' in parameters:
                postprocessing.append(
                    {
                        'type': 'esrgan',
                        'scale': int(parameters['upscale'][0]),
                        'strength': float(parameters['upscale'][1]),
                    }
                )

            rfc_dict['postprocessing'] = (
                postprocessing if len(postprocessing) > 0 else None
            )

            # semantic drift
            rfc_dict['sampler'] = parameters['sampler_name']

            # display weighted subprompts (liable to change)
            subprompts = split_weighted_subprompts(parameters['prompt'])
            subprompts = [{'prompt': x[0], 'weight': x[1]} for x in subprompts]
            rfc_dict['prompt'] = subprompts

            # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
            variations = []

            if 'with_variations' in parameters:
                variations = [
                    {'seed': x[0], 'weight': x[1]}
                    for x in parameters['with_variations']
                ]

            rfc_dict['variations'] = variations

            if 'init_img' in parameters:
                rfc_dict['type'] = 'img2img'
                rfc_dict['strength'] = parameters['strength']
                rfc_dict['fit'] = parameters['fit']  # TODO: Noncompliant
                rfc_dict['orig_hash'] = calculate_init_img_hash(
                    self.get_image_path_from_url(parameters['init_img'])
                )
                rfc_dict['init_image_path'] = parameters[
                    'init_img'
                ]  # TODO: Noncompliant
                if 'init_mask' in parameters:
                    rfc_dict['mask_hash'] = calculate_init_img_hash(
                        self.get_image_path_from_url(parameters['init_mask'])
                    )  # TODO: Noncompliant
                    rfc_dict['mask_image_path'] = parameters[
                        'init_mask'
                    ]  # TODO: Noncompliant
            else:
                rfc_dict['type'] = 'txt2img'

            metadata['image'] = rfc_dict

            return metadata

        except Exception as e:
            print('\n')
            traceback.print_exc()
            print('\n')

    def parameters_to_post_processed_image_metadata(self, parameters, original_image_path):
        try:
            current_metadata = retrieve_metadata(original_image_path)[
                'sd-metadata'
            ]
            postprocessing_metadata = {}

            """
            if we don't have an original image metadata to reconstruct,
            need to record the original image and its hash
            """
            if 'image' not in current_metadata:
                current_metadata['image'] = {}

                orig_hash = calculate_init_img_hash(
                    self.get_image_path_from_url(original_image_path)
                )

                postprocessing_metadata['orig_path'] = (original_image_path,)
                postprocessing_metadata['orig_hash'] = orig_hash

            if parameters['type'] == 'esrgan':
                postprocessing_metadata['type'] = 'esrgan'
                postprocessing_metadata['scale'] = parameters['upscale'][0]
                postprocessing_metadata['strength'] = parameters['upscale'][1]
            elif parameters['type'] == 'gfpgan':
                postprocessing_metadata['type'] = 'gfpgan'
                postprocessing_metadata['strength'] = parameters[
                    'facetool_strength'
                ]
            else:
                raise TypeError(f"Invalid type: {parameters['type']}")

            if 'postprocessing' in current_metadata['image'] and isinstance(
                current_metadata['image']['postprocessing'], list
            ):
                current_metadata['image']['postprocessing'].append(
                    postprocessing_metadata
                )
            else:
                current_metadata['image']['postprocessing'] = [
                    postprocessing_metadata
                ]

            return current_metadata

        except Exception as e:
            traceback.print_exc()
            print('\n')

    def save_result_image(
        self,
        image,
        command,
        metadata,
        output_dir,
        step_index=None,
        postprocessing=False,
    ):
        try:
            pngwriter = PngWriter(output_dir)
            prefix = pngwriter.unique_prefix()

            seed = 'unknown_seed'

            if 'image' in metadata:
                if 'seed' in metadata['image']:
                    seed = metadata['image']['seed']

            filename = f'{prefix}.{seed}'

            if step_index:
                filename += f'.{step_index}'
            if postprocessing:
                filename += f'.postprocessed'

            filename += '.png'

            path = pngwriter.save_image_and_prompt_to_png(
                image=image,
                dream_prompt=command,
                metadata=metadata,
                name=filename,
            )

            return os.path.abspath(path)

        except Exception as e:
            traceback.print_exc()
            print('\n')

    def make_unique_init_image_filename(self, name):
        try:
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            return name
        except Exception as e:
            traceback.print_exc()
            print('\n')

    def write_log_message(self, message):
        """Logs the filename and parameters used to generate or process that image to log file"""
        try:
            message = f'{message}\n'
            with open(self.log_path, 'a', encoding='utf-8') as file:
                file.writelines(message)

        except Exception as e:
            traceback.print_exc()
            print('\n')

    def get_image_path_from_url(self, url):
        """Given a url to an image used by the client, returns the absolute file path to that image"""
        try:
            if 'init-images' in url:
                return os.path.abspath(
                    os.path.join(self.init_image_path, os.path.basename(url))
                )
            elif 'mask-images' in url:
                return os.path.abspath(
                    os.path.join(self.mask_image_path, os.path.basename(url))
                )
            elif 'intermediates' in url:
                return os.path.abspath(
                    os.path.join(self.intermediate_path, os.path.basename(url))
                )
            else:
                return os.path.abspath(
                    os.path.join(self.result_path, os.path.basename(url))
                )

        except Exception as e:
            traceback.print_exc()
            print('\n')

    def get_url_from_image_path(self, path):
        """Given an absolute file path to an image, returns the URL that the client can use to load the image"""
        try:
            if 'init-images' in path:
                return os.path.join(
                    self.init_image_url, os.path.basename(path)
                )
            elif 'mask-images' in path:
                return os.path.join(
                    self.mask_image_url, os.path.basename(path)
                )
            elif 'intermediates' in path:
                return os.path.join(
                    self.intermediate_url, os.path.basename(path)
                )
            else:
                return os.path.join(self.result_url, os.path.basename(path))

        except Exception as e:
            traceback.print_exc()
            print('\n')

    def save_file_unique_uuid_name(self, bytes, name, path):
        try:
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            file_path = os.path.join(path, name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            newFile = open(file_path, 'wb')
            newFile.write(bytes)
            return file_path

        except Exception as e:
            traceback.print_exc()
            print('\n')
