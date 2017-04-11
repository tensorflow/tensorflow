/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

module tf.graph.scene.contextmenu {

/** Function that converts data to a title string. */
export interface TitleFunction {
  (data: any): string;
}

/** Function that takes action based on item clicked in the context menu. */
export interface ActionFunction {
  (elem: any, d: any, i: number): void;
}

/**
 * The interface for an item in the context menu
 */
export interface ContextMenuItem {
  title: TitleFunction;
  action: ActionFunction;
}

/**
 * Returns the event listener, which can be used as an argument for the d3
 * selection.on function. Renders the context menu that is to be displayed
 * in response to the event.
 */
export function getMenu(menu: ContextMenuItem[]) {
  let menuSelection = d3.select('.context-menu');
  // Close the menu when anything else is clicked.
  d3.select('body').on(
      'click.context', function() { menuSelection.style('display', 'none'); });

  // Function called to populate the context menu.
  return function(data, index: number): void {
    // Position and display the menu.
    let event = <MouseEvent>d3.event;
    menuSelection.style({
      'display': 'block',
      'left': (event.layerX + 1) + 'px',
      'top': (event.layerY + 1) + 'px'
    });

    // Stop the event from propagating further.
    event.preventDefault();
    event.stopPropagation();

    // Add provided items to the context menu.
    menuSelection.html('');
    let list = menuSelection.append('ul');
    list.selectAll('li')
        .data(menu)
        .enter()
        .append('li')
        .html(function(d) { return d.title(data); })
        .on('click', (d, i) => {
          d.action(this, data, index);
          menuSelection.style('display', 'none');
        });
  };
};

} // close module
