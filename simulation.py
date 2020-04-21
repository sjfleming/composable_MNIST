# Helper code for creating simulated data from numeric fields, using MNIST.
# For an example usage at the command line, try:
#
# $ python simulation.py --dir ~/Desktop/mnist/ --num 10 --speckle_noise --resize --underline_noise --data date
#
# or to try out name generation, try:
#
# $ python simulation.py --dir ~/Desktop/mnist --resize --underline_noise --spacing 0.7 --data name --num 10
#
# Date images with the same numerical suffix are written in the same handwriting.
# This is also supposedly true for names, but it doesn't seem like it.


import torch
import numpy as np
from typing import List, Union, Tuple, Dict


def junkify(img: torch.Tensor,
            resize: bool = False,
            cut_bottom: bool = False,
            speckle: bool = False,
            gaussian: bool = False,
            underlines: bool = False,
            fg_value: int = 0,
            bkg_value: int = 255,
            rand: np.random.RandomState = np.random.RandomState(seed=1234)) -> torch.Tensor:
    """Add noise to an image.

    Args:
        img: The input image
        resize: True to resize the image
        cut_bottom: True to cut a few pixels off the bottom
        speckle: True to add speckle noise (blobs)
        gaussian: True to add Gaussian noise
        underlines: True to add spotty underlines
        fg_value: Foreground value
        bkg_value: Background value
        rand: For reproducibility, pass a seeded numpy random
            number generator object.

    Returns:
        img: The noisy image, possibly of a different size.

    """

    out = img

    if resize:
        height = rand.randint(low=img.shape[-2], high=2 * img.shape[-2], size=1).item()
        width = rand.randint(low=img.shape[-1], high=3 * img.shape[-1], size=1).item()
        out = torch.ones([height, width]) * bkg_value  # blank canvas with bkg_value
        x = rand.randint(low=0, high=width - img.shape[-1] + 1)
        y = rand.randint(low=0, high=height - 28 + 1)
        out[y:y + img.shape[-2], x:x + img.shape[-1]] = img  # paste image somewhere

    if cut_bottom:
        out = out[:-10, :]

    if speckle:

        # create spots
        noise = torch.zeros(out.shape)
        n = rand.randint(low=1, high=5, size=1).item()
        for _ in range(n):
            x = rand.randint(low=0, high=out.shape[-1] - 1, size=1).item()
            y = rand.randint(low=0, high=out.shape[-2] - 1, size=1).item()
            noise[..., y:y + 2, x: x + 2] = 1.

        # smooth the spots
        if len(noise.shape) == 2:
            noise = noise.unsqueeze(0).unsqueeze(0)  # add "batch" and "channel" dims
        elif len(noise.shape) == 3:
            noise = noise.unsqueeze(1)  # add "channel" dim
        smoothed_noise = torch.nn.functional.conv2d(noise,
                                                    weight=torch.ones([1, 1, 2, 2]) * 0.25,
                                                    padding=(0, 0))

        # convert to this foreground / background scheme
        smoothed_noise = smoothed_noise * (fg_value - bkg_value) + bkg_value

        # add to image
        if bkg_value > fg_value:
            out[..., 1:, 1:] = torch.where(smoothed_noise.squeeze() < out[..., 1:, 1:],
                                           smoothed_noise.squeeze(),
                                           out[..., 1:, 1:])
        else:
            out[..., 1:, 1:] = torch.where(smoothed_noise.squeeze() > out[..., 1:, 1:],
                                           smoothed_noise.squeeze(),
                                           out[..., 1:, 1:])

    if gaussian:
        out = (out + rand.randn(out.shape[-2], out.shape[-1])
               * np.abs(bkg_value - fg_value).item() / 10.)

    if underlines:
        # create an underline
        y = rand.randint(low=out.shape[-2] - 10, high=out.shape[-2] - 1, size=1).item()
        xstart = rand.randint(low=0, high=out.shape[-1] - 1, size=1).item()
        xend = rand.randint(low=xstart, high=out.shape[-1] - 1, size=1).item()

        # add to image
        out[..., y, xstart:xend] = fg_value

    return torch.clamp(out, min=min(bkg_value, fg_value), max=max(bkg_value, fg_value))


def value_to_img(value: str,
                 dataset: Dict[str, torch.Tensor],
                 writer: Union[int, None] = None,
                 spacing: float = 0.5,
                 bkg_value: int = 255,
                 fg_value: int = 0,
                 check_input: bool = True,
                 rand: np.random.RandomState = np.random.RandomState(seed=1234)) -> torch.Tensor:
    """Convert a numerical value to an image of that value.

    Args:
        value: The integer that should be converted to an image.
            This is input as a string so that you can choose to have
            a leading zero or not, but it must be a string of numbers.
        dataset: A pre-sorted lookup of a given dataset, where
            dataset[key] is a tensor containing all instances of
            that class of image specified by the key label.
        writer: The MNIST writer whose handwriting we will use.  An
            integer value from 0 to 5421.  `None` will pick at random.
        spacing: Value that controls the blank space between digits.
            Float between 0 and 1, where 0 is tight spacing.
        bkg_value: Value for background pixels.
        fg_value: Value for foreground (digit) pixels.
        check_input: False will disable input checking for a possible
            speed-up, but it's probably negligible on a cpu.
        rand: For reproducibility, pass a seeded numpy random
            number generator object.

    Returns:
        img: An image of the hand-written value.

    """

    if check_input:
        max_writer = min([d.shape[0] for d in dataset.values()]) - 1
        if writer is None:
            writer = rand.randint(low=0, high=max_writer, size=1).item()
        else:
            assert (writer >= 0) and (writer < max_writer), \
                f'writer must be >= 0 and less than the max, which is {max_writer}'

    # create blank canvas
    width = int(28 * (1 + 0.5 * (len(value) - 1) + 0.5 * spacing * (len(value) - 1))) + 1
    img = torch.zeros((28, width))
    step = int(28 * (0.5 + 0.5 * spacing))
    ranges = zip(range(0, width - 28, step),
                 range(28, width, step))

    # paste value images
    for v, region in zip(value, list(ranges)):
        digit = dataset[v][writer, ...]
        img[:, region[0]:region[1]] = img[:, region[0]:region[1]] + digit

    img = torch.clamp(img, min=0., max=1.)
    img = img * (fg_value - bkg_value) + bkg_value

    return img


def compose_random_img(allowed_values: Union[np.ndarray, List[Union[int, str]]],
                       dataset: Dict[str, torch.Tensor],
                       writer: Union[int, None] = None,
                       rand: np.random.RandomState = np.random.RandomState(seed=1234),
                       **kwargs) -> Tuple[torch.Tensor, Union[int, str]]:
    """Compose one random image, from any of the list of allowed values.

    Args:
        allowed_values: Possible values be converted to an image.
            This is input as a string so that you can choose to have
            a leading zero or not, but it must be a string of numbers.
        dataset: A pre-sorted lookup of a given dataset, where
            dataset[key] is a tensor containing all instances of
            that class of image specified by the key label.
        writer: The MNIST writer whose handwriting we will use.  An
            integer value from 0 to 5421.  `None` will pick at random.
        rand: For reproducibility, pass a seeded numpy random
            number generator object.

    Returns:
        img: An image of the hand-written value.

    """

    value = rand.choice(allowed_values, size=1, replace=False).item()
    return (value_to_img(str(value), dataset=dataset,
                         writer=writer, rand=rand, **kwargs), value)


def random_day_img(dataset: Dict[str, torch.Tensor],
                   writer: Union[int, None] = None,
                   rand: np.random.RandomState = np.random.RandomState(seed=1234),
                   leading_zero: bool = True,
                   **kwargs) -> Tuple[torch.Tensor, int]:
    """Compose an image of a valid day."""
    value = rand.choice(list(range(1, 32)), size=1, replace=False).item()
    if leading_zero:
        s = f'{value:02d}'  # prepend a zero if the value is less than 10
    else:
        s = str(value)
    return (value_to_img(s, dataset=dataset,
                         writer=writer, rand=rand, **kwargs), value)


def random_month_img(dataset: Dict[str, torch.Tensor],
                     writer: Union[int, None] = None,
                     rand: np.random.RandomState = np.random.RandomState(seed=1234),
                     leading_zero: bool = True,
                     **kwargs) -> Tuple[torch.Tensor, int]:
    """Compose an image of a valid month."""
    value = rand.choice(list(range(1, 13)), size=1, replace=False).item()
    if leading_zero:
        s = f'{value:02d}'  # prepend a zero if the value is less than 10
    else:
        s = str(value)
    return (value_to_img(s, dataset=dataset,
                         writer=writer, rand=rand, **kwargs), value)


def random_year_img(dataset: Dict[str, torch.Tensor],
                    writer: Union[int, None] = None,
                    rand: np.random.RandomState = np.random.RandomState(seed=1234),
                    **kwargs) -> Tuple[torch.Tensor, int]:
    """Compose an image of a valid year."""
    return compose_random_img(list(range(1900, 2021)), dataset=dataset,
                              writer=writer, rand=rand, **kwargs)


def random_date_img_tuple(dataset: Dict[str, torch.Tensor],
                          writer: Union[int, None] = None,
                          rand: np.random.RandomState = np.random.RandomState(seed=1234),
                          **kwargs) -> Tuple[Tuple[torch.Tensor, int],
                                             Tuple[torch.Tensor, int],
                                             Tuple[torch.Tensor, int]]:
    """Compose a tuple of images of a valid date written by one person."""

    # choose a consistent writer
    if writer is None:
        max_writer = min([d.shape[0] for d in dataset.values()]) - 1
        writer = rand.randint(low=0, high=max_writer, size=1).item()

    # choose whether the writer prepends zeros to day and month
    leading_zero = rand.randn(1) > 0.6

    return (random_day_img(dataset=dataset, writer=writer,
                           rand=rand, leading_zero=leading_zero, **kwargs),
            random_month_img(dataset=dataset, writer=writer,
                             rand=rand, leading_zero=leading_zero, **kwargs),
            random_year_img(dataset=dataset, writer=writer, rand=rand, **kwargs))


def main():

    import argparse
    import os
    import torchvision

    parser = argparse.ArgumentParser(description='Generate handwritten dates from MNIST and save')
    parser.add_argument('--dir', type=str, default=1234, dest='dir', required=True,
                        help='output directory for saved images')
    parser.add_argument('--data', type=str, default=1234, dest='data',
                        choices=['date', 'name'], required=True,
                        help='output directory for saved images')
    parser.add_argument('--num', type=int, default=10, dest='num',
                        help='number of images')
    parser.add_argument('--spacing', type=float, default=0.5, dest='spacing',
                        help='spacing of digits in [0, 1]. tightest spacing is 0.')
    parser.add_argument('--fg_value', type=int, default=0, dest='fg_value',
                        help='value of foreground pixels')
    parser.add_argument('--bkg_value', type=int, default=255, dest='bkg_value',
                        help='value of background pixels')
    parser.add_argument('--seed', type=int, default=1234, dest='seed',
                        help='random seed for reproducibility')
    parser.add_argument('--gaussian_noise', action='store_true', default=False,
                        help='include gaussian noise')
    parser.add_argument('--underline_noise', action='store_true', default=False,
                        help='include random underline noise')
    parser.add_argument('--speckle_noise', action='store_true', default=False,
                        help='include random speckle noise')
    parser.add_argument('--resize', action='store_true', default=False,
                        help='include random image rescaling')
    parser.add_argument('--bottom_crop', action='store_true', default=False,
                        help='include random over-cropping at the bottom')
    args = parser.parse_args()

    # check
    assert os.access(args.dir, os.W_OK), f'Cannot write to specified output directory\n{args.dir}'

    # load data and sort in a way where we can pull up an image by its label
    chars = dict()
    if args.data == 'date':

        # create output directories for the images
        for folder in ['day', 'month', 'year']:
            if not os.path.isdir(os.path.join(args.dir, folder)):
                os.mkdir(os.path.join(args.dir, folder))

        data = torchvision.datasets.MNIST('../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
        for i in range(10):
            idx = (data.targets == i)
            chars[str(i)] = data.data[idx]

    elif args.data == 'name':

        # create output directories for the images
        for folder in ['firstname', 'lastname']:
            if not os.path.isdir(os.path.join(args.dir, folder)):
                os.mkdir(os.path.join(args.dir, folder))

        data = torchvision.datasets.EMNIST('../data', train=True, download=True, split='letters',
                                           transform=torchvision.transforms.ToTensor())
        for i in range(1, 27):
            idx = (data.targets == i)
            chars[chr(96 + i)] = data.data[idx].permute(0, 2, 1) * 255.

    # seed random number generators
    rand = np.random.RandomState(seed=args.seed)
    rand_junk = np.random.RandomState(seed=args.seed)

    if args.data == 'date':

        day_labels = np.zeros(args.num, dtype=int)
        month_labels = np.zeros(args.num, dtype=int)
        year_labels = np.zeros(args.num, dtype=int)

        for i in range(args.num):

            # create an example
            day, month, year = random_date_img_tuple(dataset=chars,
                                                     rand=rand,
                                                     spacing=args.spacing,
                                                     fg_value=args.fg_value,
                                                     bkg_value=args.bkg_value)

            # keep track of labels
            day_labels[i] = day[1]
            month_labels[i] = month[1]
            year_labels[i] = year[1]

            # add noise optionally
            day, month, year = (junkify(img,
                                        resize=args.resize,
                                        speckle=args.speckle_noise,
                                        underlines=args.underline_noise,
                                        gaussian=args.gaussian_noise,
                                        cut_bottom=args.bottom_crop,
                                        fg_value=args.fg_value,
                                        bkg_value=args.bkg_value,
                                        rand=rand_junk)
                                for img in [day[0], month[0], year[0]])

            # save images
            torchvision.utils.save_image(day, os.path.join(args.dir, 'day', f'day{i}.png'))
            torchvision.utils.save_image(month, os.path.join(args.dir, 'month', f'month{i}.png'))
            torchvision.utils.save_image(year, os.path.join(args.dir, 'year', f'year{i}.png'))

        # save labels
        day_labels = np.concatenate((np.expand_dims(np.arange(args.num), axis=1),
                                     np.expand_dims(day_labels, axis=1)), axis=1)
        np.savetxt(os.path.join(args.dir, 'day', 'day_labels.tsv'), day_labels,
                   delimiter='\t', header='img\tlabel', fmt='%d')
        month_labels = np.concatenate((np.expand_dims(np.arange(args.num), axis=1),
                                       np.expand_dims(month_labels, axis=1)), axis=1)
        np.savetxt(os.path.join(args.dir, 'month', 'month_labels.tsv'), month_labels,
                   delimiter='\t', header='img\tlabel', fmt='%d')
        year_labels = np.concatenate((np.expand_dims(np.arange(args.num), axis=1),
                                      np.expand_dims(year_labels, axis=1)), axis=1)
        np.savetxt(os.path.join(args.dir, 'year', 'year_labels.tsv'), year_labels,
                   delimiter='\t', header='img\tlabel', fmt='%d')

    elif args.data == 'name':

        first_labels = []
        last_labels = []

        for i in range(args.num):

            # create an example
            max_writer = min([d.shape[0] for d in chars.values()]) - 1
            writer = rand.randint(low=0, high=max_writer, size=1).item()

            import unidecode  # for removing accent marks from names

            firstnames = np.genfromtxt('../validation/first_names.all.txt', dtype=str, skip_header=1)
            firstnames = np.array([unidecode.unidecode(s.replace("'", '').replace('-', ''))
                                   for s in firstnames], dtype=str)  # remove punctuation
            firstname = compose_random_img(allowed_values=firstnames, dataset=chars,
                                           writer=writer, rand=rand, spacing=args.spacing,
                                           bkg_value=args.bkg_value, fg_value=args.fg_value)

            lastnames = np.genfromtxt('../validation/last_names.all.txt', dtype=str)
            lastnames = np.array([unidecode.unidecode(s.replace("'", '').replace('-', ''))
                                  for s in lastnames], dtype=str)  # remove punctuation
            lastname = compose_random_img(allowed_values=lastnames, dataset=chars,
                                          writer=writer, rand=rand, spacing=args.spacing,
                                          bkg_value=args.bkg_value, fg_value=args.fg_value)

            # keep track of labels
            first_labels.append(firstname[1])
            last_labels.append(lastname[1])

            # need to thin these out: the lines are too thick!
            from skimage.morphology import erosion, dilation
            if args.bkg_value > args.fg_value:
                firstname = torch.tensor(dilation(firstname[0].numpy()))
                lastname = torch.tensor(dilation(lastname[0].numpy()))
            else:
                firstname = torch.tensor(erosion(firstname[0].numpy()))
                lastname = torch.tensor(erosion(lastname[0].numpy()))

            # add noise optionally
            firstname, lastname = (junkify(img,
                                           resize=args.resize,
                                           speckle=args.speckle_noise,
                                           underlines=args.underline_noise,
                                           gaussian=args.gaussian_noise,
                                           cut_bottom=args.bottom_crop,
                                           fg_value=args.fg_value,
                                           bkg_value=args.bkg_value,
                                           rand=rand_junk)
                                   for img in [firstname, lastname])

            # save image
            torchvision.utils.save_image(firstname, os.path.join(args.dir, 'firstname', f'first{i}.png'))
            torchvision.utils.save_image(lastname, os.path.join(args.dir, 'lastname', f'last{i}.png'))

        # save labels
        first_labels = np.concatenate((np.expand_dims(np.arange(args.num), axis=1),
                                       np.expand_dims(np.array(first_labels), axis=1)), axis=1)
        np.savetxt(os.path.join(args.dir, 'firstname', 'first_labels.tsv'), first_labels,
                   delimiter='\t', header='img\tlabel', fmt='%s')
        last_labels = np.concatenate((np.expand_dims(np.arange(args.num), axis=1),
                                      np.expand_dims(last_labels, axis=1)), axis=1)
        np.savetxt(os.path.join(args.dir, 'lastname', 'last_labels.tsv'), last_labels,
                   delimiter='\t', header='img\tlabel', fmt='%s')


if __name__ == '__main__':
    main()
