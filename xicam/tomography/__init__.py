import numpy as np
from qtpy.QtWidgets import QLabel, QComboBox, QHBoxLayout, QWidget, QSpacerItem, QSizePolicy, \
    QVBoxLayout, QTreeWidget, QTreeWidgetItem, QMainWindow, QDockWidget, QPushButton, QCheckBox, QSpinBox, QStackedWidget
from xicam.core import msg

from qtpy.QtCore import Qt

from xicam.plugins import GUILayout, GUIPlugin
from xicam.gui.widgets.dynimageview import DynImageView
from xicam.gui.widgets.imageviewmixins import CatalogView
from xicam.gui.widgets.linearworkfloweditor import WorkflowEditor

import numpy as np

from xicam.plugins.operationplugin import (limits, describe_input, describe_output,
                                           operation, opts, output_names, input_names, visible)

from xicam.core.execution import Workflow
from xicam.core.data import MetaXArray

import tomopy
import dxchange

from functools import partial

@operation  # Required - defines the function below as an OperationPlugin
@input_names("path", "sinoindex", "chunksize")
@output_names("tomo", "flats", "darks", "angles")  # Required - describes the names of the output(s)
def read_ALS(path: str, sinoindex: int = 5, chunksize: int = 5) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    import h5py
    import numpy
    from math import pi
    import tomopy

    h5file = h5py.File(path, "r")
    group = list(h5file.keys())[0]
    dgroup = h5file[group]

    nangles = int(dgroup.attrs['nangles'])
    h5file.close()

    angles = tomopy.angles(nangles)

    tomo, flats, darks, _ = dxchange.read_als_832h5(path, sino=(sinoindex, sinoindex+chunksize, 1))

    print("PROCESSING SINOINDEX", tomo.shape)
    # print("HERE", tomo, flats, darks, angles)
    return tomo, flats, darks, angles

@operation  # Required - defines the function below as an OperationPlugin
@input_names("tomo", "flats", "darks")
@output_names("tomo")  # Required - describes the names of the output(s)
def tomopy_normalize(tomo : np.ndarray, flats : np.ndarray, darks : np.ndarray) -> np.ndarray:
    if flats is None or darks is None:
        return tomo

    return tomopy.normalize(tomo, flats, darks)

@operation
@input_names("tomo", "dif", "size", "axis", "ncore")
@output_names("tomo")  # Required - describes the names of the output(s)
def tomopy_remove_outlier(tomo, dif: int = 500, size: int = 5, axis=0, ncore=None):
    # outliers.dif.value = 500
    # outliers.size.value = 5
    # print("TOMO type", tomo.dtype)
    # return tomopy.remove_outlier(tomo, dif, size=size, axis=axis, ncore=ncore, out=tomo)
    return tomo

@operation
@input_names("tomo", "ncore")
@output_names("tomo")  # Required - describes the names of the output(s)
def tomopy_neg_log(tomo, ncore=None):
    print("neg log")
    tomo = tomopy.minus_log(tomo, ncore=ncore)
    return tomo

@operation
@input_names("tomo", "axis", "npad", "mode", "ncore")
@output_names("tomo")  # Required - describes the names of the output(s)
def tomopy_pad(tomo, axis=None, npad=None, mode='constant', ncore=None):
    #axis = 2
    #npad = 448
    #mode = 'edge'
    #return tomopy.pad(tomo, axis, npad=npad, mode=mode, ncore=ncore)
    return tomo

@operation
@input_names("tomo", "angles", "algorithm", 'center', 'filter_name', "nchunk", "ncore")
@output_names("recon")  # Required - describes the names of the output(s)
def tomopy_recon(tomo, angles, algorithm: str = 'gridrec', center: int = None, filter_name: str = 'butterworth', nchunk=1, ncore=None):
    #filter_name = 'butterworth'
    #algorithm = 'gridrec'
    #center = np.array([1295 + 448])  # 1295

    if center is None:
        center = np.array([tomo.shape[2]//2])
    filter_par = np.array([0.2, 2])

    num_gridx = None
    num_gridy = None

    kwargs = {
        'num_gridx': num_gridy,
        'num_gridy': num_gridx,
        'filter_name': filter_name,
        'filter_par': filter_par,
    }

    # remove unset kwargs
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # print(tomo.shape, angles.shape, center)

    print("RECONSTRUCTING with: ", algorithm)
    result = tomopy.recon(tomo, angles, center=center, algorithm=algorithm)

    # print("RETURNING", result)
    return result


class TomographyWorkflow(Workflow):
    """Example workflow that contains two operations: invert and random_noise"""
    def __init__(self):
        super(TomographyWorkflow, self).__init__(name="Tomography Workflow")

        # Create instances of our operations
        read_file = read_ALS()
        normalize = tomopy_normalize()
        remove_outlier = tomopy_remove_outlier()
        neg_log = tomopy_neg_log()
        pad = tomopy_pad()
        recon = tomopy_recon()

        self.add_operation(read_file)
        self.add_operation(normalize)
        self.add_operation(remove_outlier)
        self.add_operation(neg_log)
        self.add_operation(pad)
        self.add_operation(recon)

        self.add_link(read_file, normalize, "tomo", "tomo")
        self.add_link(read_file, normalize, "flats", "flats")
        self.add_link(read_file, normalize, "darks", "darks")

        self.add_link(normalize, remove_outlier, "tomo", "tomo")
        self.add_link(remove_outlier, neg_log, "tomo", "tomo")
        self.add_link(neg_log, pad, "tomo", "tomo")
        self.add_link(pad, recon, "tomo", "tomo")
        self.add_link(read_file, recon, "angles", "angles")

        # self.auto_connect_all()
        self._pretty_print()  # Optional - prints out the workflow visually to the console


def read_als_hdf5(filename):
    import h5py
    import numpy

    try:
        a = h5py.File(filename, "r")
        k1 = list(a.keys())
        print(k1)
        y = k1[0]  # group...

        k2 = list(a[y].keys())
        dataset = []
        bkg = []
        drk = []

        for x in k2:
            if x.find("bak_") >= 0:
                bkg.append(numpy.array(a[y][x]).T)
            elif x.find("drk_") >= 0:
                drk.append(numpy.array(a[y][x]).T)
            else:
                dataset.append(numpy.array(a[y][x]))

        dataset = numpy.vstack(dataset).squeeze().astype(numpy.float)

        if len(bkg) >= 0:
            bkg = numpy.vstack(bkg).squeeze().astype(numpy.float)

        if len(drk) >= 0:
            drk = numpy.vstack(drk).squeeze().astype(numpy.float)

        print(dataset.shape)
        dataset = numpy.swapaxes(dataset, 1, 2)
        dataset = numpy.ascontiguousarray(dataset, dtype=np.float32)
        return dataset, [], []
    except:
        print("UNABLE TO LOAD", filename)
        pass

    return [], [], []


hasLTT = False
try:
    import sys

    sys.path.append("/Users/hari/pilot/LTT_Mac_v1.6.12")
    from LTTserver import LTTserver
    hasLTT = True
except:
    pass

if hasLTT:
    @operation  # Required - defines the function below as an OperationPlugin
    @input_names("path")
    @output_names("recon")  # Required - describes the names of the output(s)
    def read_LTT(path: str) -> np.ndarray:
        import h5py
        import numpy

        print("LTT SERVER")
        ltt = LTTserver()

        dataset, bkg, drk = read_als_hdf5(path)

        a = h5py.File(path, "r")
        k1 = list(a.keys())
        group = k1[0] # group...

        k2 = list(a[group].keys())

        dataset_attrs = {}

        for key in list(a[group].attrs):
            if key.find("archdir") >= 0:
                continue

            if type(a[group].attrs[key]) == str:
                print("KEY", key, a[group].attrs[key])
                dataset_attrs[key] = a[group].attrs[key]
            else:
                print("KEY", key, a[group].attrs[key].decode(encoding='ascii'))
                dataset_attrs[key] = a[group].attrs[key].decode(encoding='ascii')

        a.close()

        if "dataType" not in dataset_attrs:
            dataset_attrs["dataType"] = "RAW_UNCALIB"

        if "projectionDataType" not in dataset_attrs:
            dataset_attrs["projectionDataType"] = "RAW_UNCALIB"

        # ltt.run("archdir = \"/Users/hari/Data/tif\"")  # <enter full path of where data is stored here>   # UPDATE
        # ltt.run("loadsct \"./\"")  # <enter path of sct file relative to archdir here> # UPDATE
        # ltt.run("evalxsize = 0")
        # ltt.run("evalysize = 0")
        # ltt.run("doBakCorr = step(evalxsize * evalysize)")

        # ltt.run("readFromDisk = step(4 * nrays * nslices * (nangles + 2 * maxThreads) / 2 ^ 30 - maxMemory)")
        ltt.run("readFromDisk = false")
        # ltt.run("writeToDisk = true")
        ltt.run("doBakCorr = false")
        ltt.run("writeToDisk = false")
        ltt.run("autoSaveSCT = false")
        ltt.run("organizeOutput = false")

        for attr in dataset_attrs:
            print(attr + " = " + dataset_attrs[attr])
            ltt.run(attr + " = " + dataset_attrs[attr])

        dataset = dataset.astype(numpy.float32)
        # ltt.run("dataType = RAW_CALIB")

        ltt.setAllProjections(dataset, 1)

        # ltt.run("defaultVolume")
        ltt.run("makeAttenRadsALS")
        # ltt.run("dataType = ATTEN_RAD")
        ltt.run("outlierCorrection")
        ltt.run("findCenter")
        ltt.run("ringRemoval")
        ltt.run("FBP")

        allSino = numpy.array(ltt.getAllSinograms())

        print("ALL Sino", allSino, allSino.shape)

        return allSino


    class LTTWorkflow(Workflow):
        def __init__(self):
            super(LTTWorkflow, self).__init__(name="LTT Workflow")

            # Create instances of our operations
            read_file = read_LTT()
            self.add_operation(read_file)

            self._pretty_print()  # Optional - prints out the workflow visually to the console


class MyCatalogView(CatalogView):
    def __init__(self):
        super().__init__()
        # parent = self.ui.roiBtn.parentWidget()

        showButton = QPushButton("Show/Hide")
        showButton.clicked.connect(self.pressed)
        self.layout().addWidget(showButton)

    def pressed(self):
        if not self.ui.roiBtn.isVisible():
            self.ui.histogram.show()
            self.ui.roiBtn.show()
            self.ui.menuBtn.show()
        else:
            self.ui.histogram.hide()
            self.ui.roiBtn.hide()
            self.ui.menuBtn.hide()

class HDFTreeViewer(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        pass

    def load(self, file_path):
        import h5py

        self.clear()

        root_item = QTreeWidgetItem(self)
        root_item.setText(0, file_path)

        def visitor_func(parent, name, node):
            # print("NAME", name, node, parent)
            parent_widget = QTreeWidgetItem(parent)

            if isinstance(node, h5py.Dataset):
                # node is a dataset
                parent_widget.setText(0, name)
                parent_widget.setText(1, "(" + ','.join([str(t) for t in node.shape ]) + ")")
            else:
                # node is a group
                parent_widget.setText(0, name)
                parent_widget.setText(1, "--")
                node.visititems(partial(visitor_func, parent_widget))

        with h5py.File(file_path, 'r') as f:
            f.visititems(partial(visitor_func, root_item))


class TomographyPlugin(GUIPlugin):
    name = "Tomography"

    def __init__(self, *args, **kwargs):
        self.active_filename = ""

        self._main_view = QMainWindow()
        self._main_view.setCentralWidget(QWidget())
        self._main_view.centralWidget().hide()

        self._editors = QStackedWidget()

        self.workflow_process_selector = QComboBox()
        self.workflow_process_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.workflow_process_selector.addItem("TomoPy Workflow")

        self._workflow = None

        self._tomo_workflow = TomographyWorkflow()  # Create a workflow
        self._workflow = self._tomo_workflow

        self._workflow_editor = WorkflowEditor(workflow=self._workflow)
        self._workflow_editor.sigRunWorkflow.connect(self.run_workflow)

        self._active_workflow_executor = QTreeWidget()

        self._editors.addWidget(self._workflow_editor)

        if hasLTT:
            self._alt_workflow = LTTWorkflow()
            self.workflow_process_selector.addItem("LTT Workflow")
            self._alt_workflow_editor = WorkflowEditor(self._alt_workflow)
            self._alt_workflow_editor.sigRunWorkflow.connect(self.run_workflow)
            self._editors.addWidget(self._alt_workflow_editor)

        self.workflow_process_selector.setCurrentIndex(0)
        self.workflow_process_selector.currentIndexChanged.connect(self.active_workflow_changed)

        self._active_workflow_widget = QWidget()
        self._active_layout = QVBoxLayout()

        self._active_layout.addWidget(self.workflow_process_selector)
        self._active_layout.addWidget(self._editors)
        self._active_layout.addWidget(self._active_workflow_executor)

        self._active_workflow_widget.setLayout(self._active_layout)

        # self.hdf5_viewer = HDFTreeViewer()
        # self.hdf5_viewer.load("/Users/hari/20200521_160002_heartbeat_test.h5")

        self.top_controls = QWidget()
        self.top_controls_layout = QHBoxLayout()
        self.top_controls.setLayout(self.top_controls_layout)
        self.top_controls_add_new_selector = QComboBox()
        self.top_controls_frequency_selector = QSpinBox()
        self.top_controls_frequency_selector.setValue(0)

        self.top_controls_add_new_viewer = QPushButton()
        self.top_controls_add_new_viewer.setText("Add Viewer")
        self.top_controls_add_new_viewer.clicked.connect(self.add_new_viewer_selected)

        self.top_controls_preview = QPushButton()
        self.top_controls_preview.setText("Preview")
        self.top_controls_preview.clicked.connect(self.preview_workflows)

        self.top_controls_execute = QPushButton()
        self.top_controls_execute.setText("Execute All")
        self.top_controls_execute.clicked.connect(self.execute_workflows)

        self.top_controls_add_new_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.top_controls_add_new_viewer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.top_controls_execute.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.top_controls_frequency_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.top_controls_layout.addWidget(self.top_controls_add_new_selector)
        self.top_controls_layout.addWidget(self.top_controls_add_new_viewer)
        self.top_controls_layout.addWidget(self.top_controls_frequency_selector)
        self.top_controls_layout.addWidget(self.top_controls_preview)
        self.top_controls_layout.addWidget(self.top_controls_execute)

        self.top_controls.setLayout(self.top_controls_layout)

        self.status_bar = QLabel("None Loaded...")

        # Create a layout to organize our widgets
        # The first argument (which corresponds to the center widget) is required.
        catalog_viewer_layout = GUILayout(self._main_view,
                                          top=self.top_controls,
                                          right=self._active_workflow_widget,
                                          bottom=self.status_bar)

        # Create a "View" stage that has the catalog viewer layout
        self.stages = {"View": catalog_viewer_layout}

        self.active_workflows = []

        # For classes derived from GUIPlugin, this super __init__ must occur at end
        super(TomographyPlugin, self).__init__(*args, **kwargs)

    def active_workflow_changed(self, index):
        self._editors.setCurrentIndex(index)

        if index == 0:
            self._workflow = self._tomo_workflow
        else:
            self._workflow = self._alt_workflow

    def preview_workflows(self):
        if len(self.active_filename) == 0:
            return

        preview_sinogram = self.top_controls_frequency_selector.value()

        workflows = []
        for aw in range(self._active_workflow_executor.topLevelItemCount()):
            widget_item = self._active_workflow_executor.topLevelItem(aw)
            workflow = widget_item.data(0, Qt.UserRole)
            workflows.append(workflow)

        active_workflows = self.active_workflows

        for workflow in workflows:
            def results_ready(workflow, *results):
                output_image = results[0]["recon"]  # We want the output_image from the last operation
                # print("RECON SHAPE", output_image.shape)
                for active_workflow in active_workflows:
                    if active_workflow[1] == workflow:
                        active_workflow[0].widget().setImage(output_image)  # Update the result view widget
            workflow.execute(callback_slot=partial(results_ready, workflow), path=self.active_filename, sinoindex=preview_sinogram)

    def execute_workflows(self):
        if len(self.active_filename) == 0:
            return

        workflows = []
        for aw in range(self._active_workflow_executor.topLevelItemCount()):
            widget_item = self._active_workflow_executor.topLevelItem(aw)
            workflow = widget_item.data(0, Qt.UserRole)
            workflows.append(workflow)

        active_workflows = self.active_workflows

        for workflow in workflows:
            def results_ready(workflow, *results):
                output_image = results[0]["recon"]  # We want the output_image from the last operation
                # print("RECON SHAPE", output_image.shape)
                for active_workflow in active_workflows:
                    if active_workflow[1] == workflow:
                        active_workflow[0].widget().setImage(output_image)  # Update the result view widget
            workflow.execute(callback_slot=partial(results_ready, workflow), path=self.active_filename)

    def add_new_viewer_selected(self, checked):
        if self.top_controls_add_new_selector.count() == 0:
            return

        selected = self.top_controls_add_new_selector.currentIndex()

        if selected < 0:
            return

        workflow = self.top_controls_add_new_selector.currentData(Qt.UserRole)
        print("WORKFLOW", workflow, workflow.name)

        dock_widget = QDockWidget()
        viewer = CatalogView()
        dock_widget.setWidget(viewer)
        dock_widget.setWindowTitle(workflow.name)

        if len(self.active_workflows) == 0:
            self._main_view.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
        else:
            self._main_view.tabifyDockWidget(self.active_workflows[-1][0], dock_widget)

        self.active_workflows.append((dock_widget, workflow))

    def appendHeader(self, header, **kwargs):
        filename = header._documents["start"][0]["filename"]
        self.active_filename = filename

        self.status_bar.setText("Loading Filename:" + filename)
        
        dataset, bkgs, drks = read_als_hdf5(filename)
        slice_widget = MyCatalogView()
        dw = QDockWidget()
        dw.setWidget(slice_widget)
        dw.setWindowTitle(filename)
        slice_widget.setImage(dataset)

        dock_children = self._main_view.findChildren(QDockWidget)
        print(dock_children)

        #if len(dock_children) == 0:
        #    self._main_view.addDockWidget(Qt.LeftDockWidgetArea, dw)
        #else:
        #    self._main_view.tabifyDockWidget(dock_children[0], dw)

        self._main_view.addDockWidget(Qt.LeftDockWidgetArea, dw)
        self.status_bar.setText("Done Loading Filename:" + filename)

    def appendCatalog(self, catalog, **kwargs):
        """Re-implemented from GUIPlugin - gives us access to a catalog reference
        You MUST implement this method if you want to load catalog data into your GUIPlugin.
        """
        # Set the catalog viewer's catalog, stream, and field (so it knows what to display)
        # This is a quick and simple demonstration; stream and field should NOT be hardcoded
        # stream = "primary"
        # field = "img"
        # self._catalog_viewer.setCatalog(catalog, stream, field)
        print("CATALOG CALLED", catalog)
        pass

    def run_workflow(self):
        """Run the internal workflowi.
        In this example, this will be called whenever the "Run Workflow" in the WorkflowEditor is clicked.
        """
        item = QTreeWidgetItem()

        workflow_name = self._workflow.name + ":" + str(self._active_workflow_executor.model().rowCount())
        workflow = self._workflow.clone()
        workflow.name = workflow_name
        workflow._pretty_print()

        item.setText(0, workflow_name)
        item.setData(0, Qt.UserRole, workflow)
        self._active_workflow_executor.addTopLevelItem(item)
        self.top_controls_add_new_selector.addItem(workflow_name, userData=workflow)
