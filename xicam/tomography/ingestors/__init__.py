import time
import h5py
import event_model
from pathlib import Path
import dask.array as da
from xarray import DataArray


def ingest_nxTOMO(paths):

    assert len(paths) == 1
    path = paths[0]

    h5 = h5py.File(path, 'r')

    data = h5['irmap']['DATA']['data']
    energy = h5['irmap']['DATA']['energy'][()]
    sample_x = h5['irmap']['DATA']['sample_x'][()]
    sample_y = h5['irmap']['DATA']['sample_y'][()]

    xarray = DataArray(data, dims=['E (eV)', 'y (μm)', 'x (μm)'], coords=[energy, sample_y, sample_x])
    dask_data = da.from_array(xarray)

    projections = [('nxSTXM',
                        {'irmap/DATA/data': ('primary', 'raw'),
                         'irmap/DATA/energy': energy,
                         'irmap/DATA/sample_x': sample_x,
                         'irmap/DATA/sample_y': sample_y}
    )]

    # Compose run start
    run_bundle = event_model.compose_run()  # type: event_model.ComposeRunBundle
    start_doc = run_bundle.start_doc
    start_doc["sample_name"] = Path(paths[0]).resolve().stem
    start_doc["projections"] = projections
    yield 'start', start_doc

    # Compose descriptor
    source = 'nxSTXM'
    frame_data_keys = {'raw': {'source': source,
                               'dtype': 'number',
                               'dims': xarray.dims,
                               # 'coords': [energy, sample_y, sample_x],
                               'shape': data.shape}}
    frame_stream_name = 'primary'
    frame_stream_bundle = run_bundle.compose_descriptor(data_keys=frame_data_keys,
                                                        name=frame_stream_name,
                                                        # configuration=_metadata(path)
                                                        )
    yield 'descriptor', frame_stream_bundle.descriptor_doc

    # NOTE: Resource document may be meaningful in the future. For transient access it is not useful
    # # Compose resource
    # resource = run_bundle.compose_resource(root=Path(path).root, resource_path=path, spec='NCEM_DM', resource_kwargs={})
    # yield 'resource', resource.resource_doc

    # Compose datum_page
    # z_indices, t_indices = zip(*itertools.product(z_indices, t_indices))
    # datum_page_doc = resource.compose_datum_page(datum_kwargs={'index_z': list(z_indices), 'index_t': list(t_indices)})
    # datum_ids = datum_page_doc['datum_id']
    # yield 'datum_page', datum_page_doc

    yield 'event', frame_stream_bundle.compose_event(data={'raw': dask_data},
                                                     timestamps={'raw': time.time()})

    yield 'stop', run_bundle.compose_stop()


if __name__ == "__main__":

    path = 'ir_stxm.h5'

    docs = list(ingest_nxSTXM([path]))

    print(docs)
