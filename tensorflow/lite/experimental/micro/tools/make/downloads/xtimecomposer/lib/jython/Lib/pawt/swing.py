"""
No longer needed, but keeping for backwards compatibility.
"""
from javax import swing
import sys

def test(panel, size=None, name='Swing Tester'):
    f = swing.JFrame(name, windowClosing=lambda event: sys.exit(0))
    if hasattr(panel, 'init'):
        panel.init()

    f.contentPane.add(panel)
    f.pack()
    if size is not None:
        from java import awt
        f.setSize(apply(awt.Dimension, size))
    f.setVisible(1)
    return f

if swing is not None:
    import pawt, sys
    pawt.swing = swing
    sys.modules['pawt.swing'] = swing
    swing.__dict__['test'] = test
